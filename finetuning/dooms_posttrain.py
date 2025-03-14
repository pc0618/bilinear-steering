import os
import torch
import numpy as np
import argparse
from datasets import load_dataset
from torch.utils.data import IterableDataset
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from torch import nn
import gc

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TokenStreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=1024, target_tokens=100_000_000):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_tokens = target_tokens
        self.tokens_seen = 0
        self.documents_seen = 0
        
    def __iter__(self):
        # Stream through the dataset
        iter_dataset = iter(self.dataset)
        
        for example in iter_dataset:
            # Process example
            text_field = 'text' if 'text' in example else 'content'
            if text_field not in example:
                # Find any text field in the example
                for key in example.keys():
                    if isinstance(example[key], str) and len(example[key]) > 0:
                        text_field = key
                        break
                    
            if text_field not in example:
                logger.warning(f"Could not find text field in example with keys: {list(example.keys())}")
                continue
                
            text = example[text_field]
            if not isinstance(text, str) or len(text) == 0:
                continue
                
            encodings = self.tokenizer(text, truncation=True, max_length=self.max_length)
            input_ids = torch.tensor(encodings['input_ids'])
            
            # Track tokens
            tokens_in_example = len(input_ids)
            self.tokens_seen += tokens_in_example
            self.documents_seen += 1
            
            # Log progress
            if self.tokens_seen % 10_000_000 < tokens_in_example or self.documents_seen % 10000 == 0:
                logger.info(f"Training: {self.tokens_seen:,}/{self.target_tokens:,} tokens "
                          f"({(self.tokens_seen/self.target_tokens)*100:.1f}%), "
                          f"{self.documents_seen:,} documents processed")
                
            if self.tokens_seen >= self.target_tokens:
                logger.info(f"Reached target token count: {self.tokens_seen:,} tokens")
                break
                
            yield {'input_ids': input_ids}

# Custom Trainer with Q-matrix regularization
class SteeringTrainer(Trainer):
    def __init__(self, 
                 model, 
                 args, 
                 train_dataset=None, 
                 eval_dataset=None, 
                 tokenizer=None, 
                 data_collator=None, 
                 compute_metrics=None, 
                 callbacks=None, 
                 optimizers=(None, None), 
                 preprocess_logits_for_metrics=None,
                 steer_layers=None,
                 lambda_reg=0.0,
                 probe_weights=None):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.steer_layers = steer_layers if steer_layers else []
        self.lambda_reg = lambda_reg
        self.probe_weights = probe_weights if probe_weights else {}
        
        # Initialize optimization: freeze all parameters except W and V matrices in specified layers
        self._freeze_selective_parameters()
        
        # Log configuration
        logger.info(f"Steering layers: {self.steer_layers}")
        logger.info(f"Lambda regularization: {self.lambda_reg}")
        logger.info(f"Number of probe weights loaded: {len(self.probe_weights)}")
        
    def _freeze_selective_parameters(self):
        """Freeze all parameters except W and V matrices in specified layers"""
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Then unfreeze W and V matrices in specified layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError("Unsupported model architecture: Cannot locate layers")
            
        for layer_idx in self.steer_layers:
            if layer_idx < len(layers):
                # Unfreeze W matrix (gate projection in MLP)
                if hasattr(layers[layer_idx].mlp, 'gate_proj'):
                    layers[layer_idx].mlp.gate_proj.weight.requires_grad = True
                    logger.info(f"Unfrozen layer {layer_idx} W matrix (gate_proj)")
                
                # Unfreeze V matrix (up projection in MLP)
                if hasattr(layers[layer_idx].mlp, 'up_proj'):
                    layers[layer_idx].mlp.up_proj.weight.requires_grad = True
                    logger.info(f"Unfrozen layer {layer_idx} V matrix (up_proj)")
            else:
                logger.warning(f"Layer index {layer_idx} out of range (max: {len(layers)-1})")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss with Q-matrix regularization for specified layers
        """
        # Regular loss computation
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add regularization term if lambda > 0
        if self.lambda_reg > 0 and self.steer_layers and self.probe_weights:
            reg_loss = self._compute_q_matrix_regularization()
            total_loss = loss + self.lambda_reg * reg_loss
            
            # Log regularization loss occasionally
            if self.state.global_step % 100 == 0:
                logger.info(f"Step {self.state.global_step}: Base loss: {loss.item():.4f}, "
                           f"Reg loss: {reg_loss.item():.4f}, "
                           f"Reg contribution: {(self.lambda_reg * reg_loss).item():.4f}, "
                           f"Total: {total_loss.item():.4f}")
                
                # Try to log to wandb if available
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "loss/base": loss.item(),
                            "loss/reg_raw": reg_loss.item(),
                            "loss/reg_scaled": (self.lambda_reg * reg_loss).item(),
                            "loss/total": total_loss.item(),
                            "loss/reg_percentage": (self.lambda_reg * reg_loss).item() / total_loss.item() * 100,
                            "step": self.state.global_step
                        })
                except:
                    pass
            
            # Return the total loss (base + regularization)
            loss = total_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_q_matrix_regularization(self):
        """
        Compute the regularization term based on the Frobenius norm of Q matrices
        
        According to the formula:
        Q = Sum(u_a * B_a) where B_a is the bilinear tensor contracted with output direction u
        
        For TinyLlama bilinear MLP:
        - W (gate_proj): [hidden_dim, input_dim] mapping from input to hidden
        - V (up_proj): [hidden_dim, input_dim] mapping from input to hidden
        - P (down_proj): [output_dim, hidden_dim] mapping from hidden to output
        """
        reg_loss = torch.tensor(0.0, device=self.model.device, dtype=torch.bfloat16)
        all_frob_norms = {}
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            return reg_loss
        
        for layer_idx in self.steer_layers:
            if layer_idx < len(layers) and layer_idx in self.probe_weights:
                # Get probe weights (output direction u) and convert to bfloat16
                u = self.probe_weights[layer_idx].to(device=self.model.device, dtype=torch.bfloat16)
                
                # Get the necessary matrices (W, V, P)
                if (hasattr(layers[layer_idx].mlp, 'gate_proj') and 
                    hasattr(layers[layer_idx].mlp, 'up_proj') and
                    hasattr(layers[layer_idx].mlp, 'down_proj')):
                    
                    W = layers[layer_idx].mlp.gate_proj.weight  # Shape: [hidden_dim, input_dim]
                    V = layers[layer_idx].mlp.up_proj.weight    # Shape: [hidden_dim, input_dim]
                    P = layers[layer_idx].mlp.down_proj.weight  # Shape: [output_dim, hidden_dim]
                    
                    # Log dimensions for debugging
                    if self.state.global_step % 100 == 0:
                        logger.info(f"Layer {layer_idx} dimensions: u={u.shape}, W={W.shape}, V={V.shape}, P={P.shape}")
                        logger.info(f"Layer {layer_idx} dtypes: u={u.dtype}, W={W.dtype}, V={V.dtype}, P={P.dtype}")
                    
                    # For TinyLlama:
                    # u has shape [output_dim] - probe weights for output feature
                    # P has shape [output_dim, hidden_dim] - down projection from hidden to output
                    # W has shape [hidden_dim, input_dim] - up projection from input to hidden
                    # V has shape [hidden_dim, input_dim] - up projection from input to hidden
                    
                    # Calculate u_P = u * P to get the effective output direction in hidden space
                    # u: [output_dim], P: [output_dim, hidden_dim] → u_P: [hidden_dim]
                    u_P = torch.matmul(u, P)  # Project the output direction to the hidden space
                    
                    # Compute Q matrix using the bilinear tensor formula from the paper
                    # Q will have shape [input_dim, input_dim]
                    Q = torch.zeros((W.shape[1], V.shape[1]), device=self.model.device, dtype=torch.bfloat16)
                    
                    # Sum up the outer products of corresponding rows of W and V, weighted by u_P
                    for i in range(u_P.shape[0]):  # Iterate over hidden dimensions
                        if u_P[i] != 0:  # Skip computations for zero weights
                            # W[i] and V[i] are the i-th rows
                            # Compute their outer product and scale by u_P[i]
                            out_prod = torch.outer(W[i], V[i])
                            Q += u_P[i] * out_prod
                    
                    # Compute the Frobenius norm
                    frob_norm = torch.norm(Q, p='fro')
                    frob_norm_squared = frob_norm**2
                    
                    # Store the norm for logging
                    all_frob_norms[f"layer_{layer_idx}"] = frob_norm.item()
                    
                    # Add to the regularization loss
                    reg_loss += frob_norm_squared
                    
                    # Log the norm occasionally
                    if self.state.global_step % 100 == 0:
                        logger.info(f"Layer {layer_idx} Q-matrix Frobenius norm: {frob_norm.item():.4f}")
                else:
                    logger.warning(f"Layer {layer_idx} missing required matrices for Q computation")
        
        # Log all Frobenius norms to wandb
        try:
            import wandb
            if wandb.run is not None:
                # Create a dict for all metrics
                metrics = {}
                
                # Add Frobenius norms for each layer
                for layer_key, norm_value in all_frob_norms.items():
                    metrics[f"frob_norm/{layer_key}"] = norm_value
                
                # Add total Frobenius norm (sum of all layers)
                if all_frob_norms:
                    metrics["frob_norm/total"] = sum(all_frob_norms.values())
                
                # Add regularization loss
                metrics["loss/reg_component"] = reg_loss.item()
                
                # Add global step
                metrics["step"] = self.state.global_step
                
                # Log all metrics
                wandb.log(metrics)
        except Exception as e:
            logger.warning(f"Error logging to wandb: {e}")
        
        return reg_loss

def load_probe_weights(probe_dir, layer_indices):
    """
    Load probe weights for the specified layers
    """
    probe_weights = {}
    
    for layer_idx in layer_indices:
        probe_path = os.path.join(probe_dir, f"mlp_out_{layer_idx}_probe.pt")
        
        if os.path.exists(probe_path):
            try:
                probe = torch.load(probe_path, map_location="cpu")
                
                # Print out probe structure for debugging
                logger.info(f"Probe for layer {layer_idx} type: {type(probe)}")
                if isinstance(probe, dict):
                    logger.info(f"Probe keys: {list(probe.keys())}")
                
                # Extract the weights - handle different possible formats
                weights = None
                if isinstance(probe, dict):
                    if "weight" in probe:
                        weights = probe["weight"]
                    elif "linear.weight" in probe:
                        weights = probe["linear.weight"]
                    elif "classifier.weight" in probe:
                        weights = probe["classifier.weight"]
                    elif "model.weight" in probe:
                        weights = probe["model.weight"]
                    elif "model" in probe and isinstance(probe["model"], dict):
                        if "weight" in probe["model"]:
                            weights = probe["model"]["weight"]
                    else:
                        # Try common tensor names
                        for key in ["weights", "probe", "linear", "classifier"]:
                            if key in probe:
                                weights = probe[key]
                                break
                        
                        # If still not found, try first tensor
                        if weights is None:
                            for k, v in probe.items():
                                if isinstance(v, torch.Tensor):
                                    weights = v
                                    logger.info(f"Using tensor from key '{k}'")
                                    break
                elif isinstance(probe, torch.Tensor):
                    weights = probe
                
                if weights is None:
                    logger.warning(f"Could not extract weights from probe for layer {layer_idx}")
                    continue
                
                # Ensure weights is a 1D tensor (output direction)
                if len(weights.shape) > 1:
                    if weights.shape[0] == 1:
                        # If it's a single row, use it directly
                        weights = weights.squeeze(0)
                    else:
                        # For 2D weights, take the first dimension
                        logger.info(f"Using first dimension of weights with shape {weights.shape}")
                        weights = weights[0]
                
                # Keep the weights in CPU for now to avoid memory issues
                # We'll move to device and convert to bfloat16 right before use
                probe_weights[layer_idx] = weights
                logger.info(f"Loaded probe weights for layer {layer_idx} with shape {weights.shape} and dtype {weights.dtype}")
            except Exception as e:
                logger.error(f"Error loading probe for layer {layer_idx}: {e}")
                logger.error(f"Exception details: {str(e)}")
        else:
            logger.warning(f"Probe file not found for layer {layer_idx}: {probe_path}")
    
    return probe_weights

def main():
    parser = argparse.ArgumentParser(description="Bilinear TinyLlama Steering Experiment")
    
    # Model and training parameters
    parser.add_argument("--model_path", type=str, default="./tinyllama-1.1b-bilinear-dooms-final",
                        help="Path to the bilinear model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./tinyllama-1.1b-steered",
                        help="Directory to save the steered model")
    parser.add_argument("--probe_dir", type=str, default="./linear_probes",
                        help="Directory containing probe weights")
    parser.add_argument("--steer_layers", type=str, default="0,1,2",
                        help="Comma-separated list of layer indices to steer")
    parser.add_argument("--lambda_reg", type=float, default=0.03,
                        help="Regularization strength for Q-matrix norm")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="Batch size for training")
    parser.add_argument("--target_tokens", type=int, default=100_000_000,
                        help="Number of tokens to train on")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Number of steps between checkpoints")
    
    args = parser.parse_args()
    
    # Parse layer indices
    steer_layers = [int(idx.strip()) for idx in args.steer_layers.split(",") if idx.strip()]
    
    # Initialize Weights & Biases
    try:
        import wandb
        wandb.login()
        wandb.init(
            project="tinyllama-dooms-posttraining",
            name=f"steering-layers-{args.steer_layers}-lambda-{args.lambda_reg}",
            config={
                "model_path": args.model_path,
                "steer_layers": steer_layers,
                "lambda_reg": args.lambda_reg,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "target_tokens": args.target_tokens,
            }
        )
        logger.info("Successfully initialized Weights & Biases monitoring.")
    except Exception as e:
        logger.warning(f"Could not initialize Weights & Biases: {e}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Make sure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load probe weights for specified layers
    logger.info(f"Loading probe weights from {args.probe_dir}")
    probe_weights = load_probe_weights(args.probe_dir, steer_layers)
    
    # Check if any probe weights were loaded successfully
    if not probe_weights:
        logger.error("No probe weights were loaded. Cannot proceed with steering experiment.")
        return
    
    # Load FineWeb dataset in streaming mode
    logger.info("Loading FineWeb dataset with streaming...")
    raw_dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    
    # Create streaming dataset with token limit
    train_dataset = TokenStreamingDataset(
        raw_dataset,
        tokenizer,
        max_length=args.max_length,
        target_tokens=args.target_tokens
    )
    
    # Estimate training steps
    tokens_per_batch = args.batch_size * args.gradient_accumulation_steps * args.max_length
    estimated_steps = args.target_tokens // tokens_per_batch
    logger.info(f"Estimated training steps: {estimated_steps:,}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_steps=estimated_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=8,  # Using 8 workers as requested
        report_to="wandb",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize custom trainer with Q-matrix regularization
    trainer = SteeringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        steer_layers=steer_layers,
        lambda_reg=args.lambda_reg,
        probe_weights=probe_weights,
    )
    
    # Start training
    logger.info("Starting training with Q-matrix regularization...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Also save a summary of the steering experiment
    with open(os.path.join(args.output_dir, "steering_info.txt"), "w") as f:
        f.write(f"Steering Experiment Summary\n")
        f.write(f"=========================\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Steered layers: {args.steer_layers}\n")
        f.write(f"Lambda regularization: {args.lambda_reg}\n")
        f.write(f"Training tokens: {args.target_tokens}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()