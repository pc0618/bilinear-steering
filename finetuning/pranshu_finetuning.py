import os
import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from einops import einsum
import numpy as np
import gc
from torch import nn
import logging
import math
import wandb
import argparse
from datetime import datetime

# Importing the Sight class from the modified utils.py
from utils import Sight

# Import the implementation from the provided code
def exp_decay(x, k=1.05):
    return min(x * np.e ** (k - k * x), 1.0)

class Interpolator(nn.Module):
    def __init__(self, original, approximation) -> None:
        super().__init__()
        self.original = original
        
        self.linear = nn.Linear(*approximation.shape, bias=False)
        self.alpha = 0.0
        
        if approximation is not None:
            self.linear.weight = nn.Parameter(approximation.detach())
    
    def forward(self, x):
        return (1 - self.alpha) * self.original(x) + self.alpha * self.linear(x)

class GateReplacer:
    def extract(self, sight, batch):
        with sight.trace(batch["input_ids"], scan=False, validate=False):
            # Check if the model is Llama-style or GPT-style
            if hasattr(sight._envoy, 'model'):
                # Llama models use a different architecture
                # In Llama, SwiGLU implementation typically uses:
                # - gate_proj for the gate projection
                # - act_fn for the SiLU activation
                g_inp = []
                g_out = []
                
                for layer in sight._envoy.model.layers:
                    # First try with act_fn (activation function)
                    if hasattr(layer.mlp, 'act_fn'):
                        g_inp.append(layer.mlp.act_fn.input.save())
                        g_out.append(layer.mlp.act_fn.output.save())
                    # Fallback - try to find SiLU or activation in the forward path 
                    else:
                        # Print layer structure to debug
                        print(f"Layer structure: {dir(layer.mlp)}")
                        # This is a guess - would need to adapt based on actual structure
                        try:
                            # Try gate_proj output for input and direct activation for output
                            g_inp.append(layer.mlp.gate_proj.output.save())
                            # For output, try to find SiLU activation inside the forward method
                            # This might need to be modified based on actual execution
                            g_out.append(layer.mlp.output.save()) 
                        except AttributeError as e:
                            raise AttributeError(f"Could not access gate mechanism in Llama model: {e}")
                
            elif hasattr(sight._envoy, 'transformer'):
                # Original GPT-style architecture
                g_inp = [layer.mlp.w.gate.input.save() for layer in sight._envoy.transformer.h]
                g_out = [layer.mlp.w.gate.output.save() for layer in sight._envoy.transformer.h]
            else:
                raise AttributeError("Model architecture not supported")
                
        return torch.stack(g_inp), torch.stack(g_out)
        
    def overwrite(self, model, x):
        # Check if the model is Llama-style or GPT-style
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama model
            for layer, x1 in zip(model.model.layers, x):
                # First try direct act_fn replacement
                if hasattr(layer.mlp, 'act_fn'):
                    # Replace the activation function
                    original_act_fn = layer.mlp.act_fn
                    layer.mlp.act_fn = Interpolator(original_act_fn, x1)
                else:
                    # This implementation may need to be adapted based on the TinyLlama structure
                    # We might need to monkey-patch the forward method or find where SiLU is applied
                    raise NotImplementedError(
                        "Direct activation function replacement not supported for this model. "
                        "Please inspect the model's forward pass to determine where to inject the Interpolator."
                    )
                
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # Original GPT-style architecture
            for layer, x1 in zip(model.transformer.h, x):
                layer.mlp.w.gate = Interpolator(layer.mlp.w.gate, x1)

def _regression_metrics(a, b, x):
    b_pred = einsum(a, x, "... b i, ... i o -> ... b o")
    residuals = b - b_pred

    ss_total = (b - b.mean(-1, keepdim=True)).pow(2).sum()
    ss_residual = residuals.pow(2).sum()
    r_squared = 1 - (ss_residual / ss_total)
    
    return dict(r_squared=r_squared.item())

def replace_components(model, tokenizer, dataset_or_loader, which="gate", n_batches=1, compute_metrics=True):
    """
    Replace components in the model with Interpolator modules.
    
    Args:
        model: The model to modify
        tokenizer: The tokenizer for the model
        dataset_or_loader: Either a dataset or a dataloader
        which: Which component to replace ('gate' for gated MLP)
        n_batches: Number of batches to use for computing the regression
        compute_metrics: Whether to compute and print regression metrics
    """
    # Fix: Pass tokenizer explicitly to Sight
    replacer = GateReplacer()
    sight = Sight(model, tokenizer=tokenizer)
    
    # Determine if input is already a dataloader or a dataset
    if isinstance(dataset_or_loader, torch.utils.data.DataLoader):
        loader = dataset_or_loader
    else:
        loader = torch.utils.data.DataLoader(
            dataset_or_loader, 
            batch_size=128, 
            shuffle=False
        )
    
    a, b = [], []
    
    # Process batches
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
            
        # Move to the right device
        if 'input_ids' in batch and hasattr(model, 'device'):
            batch['input_ids'] = batch['input_ids'].to(model.device)
            
        # Extract input and output activations
        inp, out = replacer.extract(sight, batch)
        a.append(inp)
        b.append(out)
    
    # Concatenate and reshape
    a = torch.cat(a, dim=1).flatten(1, 2)
    b = torch.cat(b, dim=1).flatten(1, 2)
    
    # Compute linear regression (for each layer's gate)
    x = torch.linalg.lstsq(a, b).solution
    
    # Compute and print regression metrics
    if compute_metrics:
        metrics = _regression_metrics(a, b, x)
        logging.info(f"Regression metrics: {metrics}")
        
        # Log regression metrics to wandb
        if wandb.run is not None:
            wandb.log({"regression/r_squared": metrics["r_squared"]})
        
    # Clean up memory
    del a, b
    gc.collect()
    torch.cuda.empty_cache()

    # Replace gates with interpolators
    replacer.overwrite(model, x)
    return model

def _set_alpha(module, alpha):
    if isinstance(module, Interpolator):
        module.alpha = alpha

class AlphaDecay(TrainerCallback):
    def __init__(self, model, total_steps, interpolation_steps, decay=exp_decay):
        """
        Custom callback for alpha decay following the paper's approach.
        
        Args:
            model: The model with Interpolator modules
            total_steps: Total number of training steps
            interpolation_steps: Number of steps for linear interpolation (30% of total)
            decay: The decay function to use after interpolation phase
        """
        self.model = model
        self.total_steps = total_steps
        self.interpolation_steps = interpolation_steps
        self.decay = decay
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
 
    def on_step_begin(self, args, state, control, **kwargs):
        current_step = state.global_step
        
        if current_step <= self.interpolation_steps:
            # Phase 1: Linear interpolation during first 30% of steps (as described in paper)
            alpha = current_step / self.interpolation_steps
        else:
            # Phase 2: Fine-tuning with full bilinear for remaining 70%
            alpha = 1.0
            
        # Apply alpha to all Interpolator modules
        self.model.apply(lambda x: _set_alpha(x, alpha))
        
        # Calculate current beta value (beta = 1 - alpha)
        beta = 1.0 - alpha
        
        # Log alpha and beta values to wandb
        if wandb.run is not None:
            wandb.log({
                "alpha": alpha,
                "beta": beta,
                "step": current_step
            }, step=current_step)
        
        # Log occasionally to console
        if current_step % 1000 == 0 or current_step == 1:
            self.logger.info(f"Step {current_step}/{self.total_steps}: alpha = {alpha:.4f}, beta = {beta:.4f}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}
            
        current_step = state.global_step
        if current_step <= self.interpolation_steps:
            alpha = current_step / self.interpolation_steps
        else:
            alpha = 1.0
            
        logs['alpha'] = alpha
        logs['beta'] = 1.0 - alpha

# Create a streaming dataset to efficiently handle large datasets
class TokenStreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=512, target_tokens=500_000_000, 
                 download_limit=1_000_000_000):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_tokens = target_tokens
        self.download_limit = download_limit
        self.tokens_seen = 0
        self.documents_seen = 0
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single worker case
            iter_dataset = iter(self.dataset)
        else:
            # Multiple workers not well supported with token counting
            # For distributed training, would need a more sophisticated approach
            self.logger.warning("Multiple workers detected. Token counting may be approximate.")
            iter_dataset = iter(self.dataset)
        
        download_tokens = 0
        
        for example in iter_dataset:
            # First, check if we've downloaded enough data
            if download_tokens >= self.download_limit:
                self.logger.info(f"Reached download limit: {download_tokens} tokens")
                break
                
            # Process the example - tokenize to count tokens
            text = example['text']
            encodings = self.tokenizer(text, truncation=True, max_length=self.max_length)
            input_ids = torch.tensor(encodings['input_ids'])
            
            # Track tokens downloaded
            document_tokens = len(input_ids)
            download_tokens += document_tokens
            self.documents_seen += 1
            
            # Check if we've seen enough tokens for training
            if self.tokens_seen >= self.target_tokens:
                # We've seen enough tokens, stop yielding but continue downloading
                # to reach download limit (to save for potential future use)
                continue
                
            # Track tokens used for training
            self.tokens_seen += document_tokens
            
            # Log progress occasionally
            if self.tokens_seen % 10_000_000 == 0 or self.documents_seen % 10000 == 0:
                self.logger.info(f"Training: {self.tokens_seen:,}/{self.target_tokens:,} tokens "
                                f"({(self.tokens_seen/self.target_tokens)*100:.1f}%)")
                self.logger.info(f"Downloaded: {download_tokens:,}/{self.download_limit:,} tokens "
                                f"({(download_tokens/self.download_limit)*100:.1f}%)")
                self.logger.info(f"Documents processed: {self.documents_seen:,}")
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        "dataset/training_tokens": self.tokens_seen,
                        "dataset/training_percentage": (self.tokens_seen/self.target_tokens)*100,
                        "dataset/downloaded_tokens": download_tokens,
                        "dataset/downloaded_percentage": (download_tokens/self.download_limit)*100,
                        "dataset/documents_seen": self.documents_seen
                    })
                
            yield {'input_ids': input_ids}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Finetune TinyLlama to a bilinear variant")
    parser.add_argument("--wandb_project", type=str, default="tinyllama-bilinear", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (team) name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./final_bilinear_tinyllama", help="Output directory for the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total_tokens", type=int, default=500_000_000, help="Total tokens to use for training")
    parser.add_argument("--download_tokens", type=int, default=500_000_000, help="Total tokens to download from FineWeb")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("finetune_bilinear.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Initialize Weights & Biases
    run_name = args.wandb_name if args.wandb_name else f"tinyllama-bilinear-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logger.info(f"Initializing wandb with project: {args.wandb_project}, run name: {run_name}")
    
    # Define token usage for this run
    total_tokens = args.total_tokens  # Use 500M tokens for training
    download_tokens = args.download_tokens  # Download 1B tokens from FineWeb
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model": "TinyLlama-1.1B",
            "total_tokens": total_tokens,
            "download_tokens": download_tokens,
            "interpolation_schedule": "linear_30_percent",
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": args.lr,
            "seed": args.seed
        }
    )
    
    logger.info("Starting bilinear finetuning process")
    
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    logger.info(f"Loading model: {model_name}")
    
    # Log model information to wandb
    wandb.config.update({"model_name": model_name})
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    # Log model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model loaded. Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params
    })
    
    # Make sure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info("Model and tokenizer loaded successfully")
    
    # Load FineWeb data as a streaming dataset to handle the large dataset efficiently
    logger.info("Loading FineWeb dataset")
    
    # Use the specified token counts
    logger.info(f"Training on {total_tokens:,} tokens, downloading {download_tokens:,} tokens total")
    
    # Load the FineWeb dataset from HuggingFace
    logger.info("Loading HuggingFaceFW/fineweb dataset")
    try:
        raw_dataset = load_dataset(
            "HuggingFaceFW/fineweb", 
            split="train", 
            streaming=True
        )
        logger.info("Successfully loaded FineWeb dataset")
    except Exception as e:
        logger.error(f"Error loading FineWeb dataset: {str(e)}")
        logger.info("Falling back to manishiitg/fineweb_english dataset")
        # Fallback to alternative dataset if needed
        raw_dataset = load_dataset(
            "manishiitg/fineweb_english", 
            split="train", 
            streaming=True
        )
    
    # Create streaming dataset with token counting
    train_dataset = TokenStreamingDataset(
        raw_dataset, 
        tokenizer,
        max_length=512,
        target_tokens=total_tokens,
        download_limit=download_tokens
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    logger.info("Dataset preparation complete")
    
    # Calculate training parameters
    batch_size = args.batch_size
    gradient_accumulation_steps = args.grad_accum
    tokens_per_batch = batch_size * 512 * gradient_accumulation_steps
    total_steps = total_tokens // tokens_per_batch
    
    # 30% of steps for linear interpolation toward β=0 as per the paper
    interpolation_steps = int(0.3 * total_steps)
    
    # Log training parameters to wandb
    wandb.config.update({
        "tokens_per_batch": tokens_per_batch,
        "total_steps": total_steps,
        "interpolation_steps": interpolation_steps,
        "interpolation_percentage": 30.0
    })
    
    logger.info(f"Training configuration: {total_tokens} tokens, {total_steps} steps")
    logger.info(f"First {interpolation_steps} steps (30%) will linearly interpolate β to 0")
    
    # Get a small sample of the dataset for the replacement procedure
    logger.info("Preparing sample data for gate replacement")
    # First, collect a small sample into a list to make it have a length
    sample_data = list(raw_dataset.take(1000))
    
    # Create a simple dataset with __len__ 
    class SampleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, tokenizer):
            self.samples = samples
            self.tokenizer = tokenizer
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            # Return tokenized text
            return {'text': self.samples[idx]['text']}
    
    sample_dataset = SampleDataset(sample_data, tokenizer)
    
    # Now create a DataLoader with this dataset
    sample_loader = torch.utils.data.DataLoader(
        sample_dataset, 
        batch_size=128, 
        shuffle=False,
        collate_fn=lambda examples: {'input_ids': tokenizer([ex['text'] for ex in examples], 
                                                           return_tensors="pt", 
                                                           padding=True, 
                                                           truncation=True, 
                                                           max_length=512).input_ids}
    )
    
    # Log sample data information
    wandb.config.update({"gate_replacement_sample_size": 1000})
    
    # Replace the gate components with interpolators
    logger.info("Replacing gate components with interpolators")
    model = replace_components(
        model,
        tokenizer,  # Fix: Pass tokenizer explicitly
        sample_loader, 
        which="gate", 
        n_batches=5,
        compute_metrics=True
    )
    logger.info("Gate replacement complete")
    
    # Configure training arguments
    logger.info("Setting up training arguments")
    output_dir = args.output_dir
    results_dir = f"{output_dir}/results"
    
    # Calculate steps based on our reduced token count (500M)
    batch_size = args.batch_size
    gradient_accumulation_steps = args.grad_accum
    tokens_per_batch = batch_size * 512 * gradient_accumulation_steps
    total_steps = total_tokens // tokens_per_batch
    
    # 30% of steps for linear interpolation toward β=0 as per the paper
    interpolation_steps = int(0.3 * total_steps)
    
    logger.info(f"Training configuration: {total_tokens:,} tokens, {total_steps:,} steps")
    logger.info(f"First {interpolation_steps:,} steps (30%) will linearly interpolate β to 0")
    
    # Log training parameters to wandb
    wandb.config.update({
        "tokens_per_batch": tokens_per_batch,
        "total_steps": total_steps,
        "interpolation_steps": interpolation_steps,
        "interpolation_percentage": 30.0
    })
    
    training_args = TrainingArguments(
        output_dir=results_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # We'll use max_steps instead
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=total_steps,
        save_steps=min(5000, total_steps // 10),  # Save at least 10 checkpoints
        save_total_limit=5,  # Keep a few checkpoints
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        fp16=True,
        learning_rate=args.lr,
        warmup_steps=int(0.01 * total_steps),  # 1% warmup
        report_to=["wandb", "tensorboard"],
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Important for our custom dataset
    )
    
    # Create custom evaluation callback to track perplexity
    class EvaluationCallback(TrainerCallback):
        def __init__(self, eval_dataset, tokenizer, model, eval_steps=2000):
            self.eval_dataset = eval_dataset
            self.tokenizer = tokenizer
            self.model = model
            self.eval_steps = eval_steps
            self.logger = logging.getLogger(__name__)
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.eval_steps == 0:
                self.logger.info(f"Performing evaluation at step {state.global_step}")
                
                # Create a small evaluation batch
                eval_samples = list(self.eval_dataset.take(64))
                eval_texts = [sample['text'] for sample in eval_samples]
                
                # Tokenize
                encodings = self.tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                input_ids = encodings.input_ids.to(model.device)
                attention_mask = encodings.attention_mask.to(model.device)
                
                # Evaluate
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss.item()
                    perplexity = torch.exp(torch.tensor(loss)).item()
                
                # Log metrics
                metrics = {
                    "eval/loss": loss,
                    "eval/perplexity": perplexity
                }
                
                self.logger.info(f"Evaluation results: Loss = {loss:.4f}, Perplexity = {perplexity:.4f}")
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
    
    # Initialize trainer with custom callbacks
    logger.info("Initializing trainer with callbacks")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[
            AlphaDecay(
                model, 
                total_steps=total_steps, 
                interpolation_steps=interpolation_steps
            ),
            EvaluationCallback(
                eval_dataset=raw_dataset,
                tokenizer=tokenizer,
                model=model,
                eval_steps=min(5000, total_steps // 20)  # Evaluate at least 20 times during training
            )
        ]
    )
    
    # Start training
    logger.info("Starting training process")
    trainer.train()
    
    # Save the final model
    logger.info("Training completed. Saving final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Additionally save as torch checkpoint
    torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
    
    # Create a config note about the model
    config_info = {
        "model_name": "TinyLlama-1.1B finetuned with bilinear MLP",
        "original_model": model_name,
        "total_tokens": total_tokens,
        "interpolation_schedule": "Linearly interpolate β to 0 during first 30% of tokens, then finetune with β=0",
        "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wandb_run_id": wandb.run.id if wandb.run else None
    }
    
    # Save as text file
    with open(f"{output_dir}/bilinear_config.txt", "w") as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
            
    # Also save in a format that can be easily loaded
    torch.save(config_info, f"{output_dir}/bilinear_config.pt")
    
    # Log model artifact to wandb
    if wandb.run is not None:
        # Create a metadata file
        with open(f"{output_dir}/metadata.json", "w") as f:
            import json
            json.dump(config_info, f, indent=2)
            
        # Log the final model as a wandb artifact
        model_artifact = wandb.Artifact(
            name=f"bilinear-tinyllama-{wandb.run.id}", 
            type="model", 
            description="TinyLlama-1.1B finetuned with bilinear MLP"
        )
        model_artifact.add_dir(output_dir)
        wandb.log_artifact(model_artifact)
    
    logger.info(f"Model successfully saved to {output_dir}")
    logger.info("Finetuning process complete!")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log the exception to wandb if it's initialized
        if wandb.run is not None:
            wandb.log({"error": str(e)})
            wandb.finish(exit_code=1)
            
        # Re-raise the exception
        raise