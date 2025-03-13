import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import math
import wandb
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bilinear_finetuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TokenStreamingDataset(IterableDataset):
    """
    Dataset for streaming tokens from a large text corpus.
    Tracks token count to ensure we don't exceed the target.
    """
    def __init__(self, dataset, tokenizer, max_length=1024, target_tokens=500_000_000):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_tokens = target_tokens
        self.tokens_seen = 0
        self.documents_seen = 0
        
    def __iter__(self):
        # Single GPU case
        iter_dataset = iter(self.dataset)
        
        for example in iter_dataset:
            # Process example
            text = example['text']
            encodings = self.tokenizer(text, truncation=True, max_length=self.max_length)
            input_ids = torch.tensor(encodings['input_ids'])
            
            # Track tokens
            self.tokens_seen += len(input_ids)
            self.documents_seen += 1
            
            # Log progress
            if self.tokens_seen % 10_000_000 == 0 or self.documents_seen % 10000 == 0:
                logger.info(f"Training: {self.tokens_seen:,}/{self.target_tokens:,} tokens "
                          f"({(self.tokens_seen/self.target_tokens)*100:.1f}%)")
                
            if self.tokens_seen >= self.target_tokens:
                logger.info(f"Reached target token count: {self.tokens_seen}")
                break
                
            yield {'input_ids': input_ids}

class InterpolatedSiLU(nn.Module):
    """
    Custom activation function that interpolates between SiLU (Swish) and a bilinear form.
    When alpha=1, acts exactly like SiLU
    When alpha=0, uses only the bilinear multiplication
    
    Note: For this finetuning, we're assuming the model is already in bilinear mode
    with alpha=0, and we'll keep it that way.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.alpha = 0.0  # Start with pure bilinear (assuming checkpoint is already bilinear)
        # Initialize bilinear weights to identity-like behavior
        self.bilinear = nn.Linear(hidden_size, hidden_size, bias=False)
        # Initialize to approximate identity operation
        nn.init.eye_(self.bilinear.weight)
        
    def forward(self, x):
        """
        Forward pass that uses bilinear transformation
        """
        # Since alpha=0, we'll just use the bilinear activation
        # Keeping the full formula for completeness and future flexibility
        silu_activation = F.silu(x)
        
        # Ensure bilinear.weight has the same dtype as x
        if self.bilinear.weight.dtype != x.dtype:
            self.bilinear.weight.data = self.bilinear.weight.data.to(dtype=x.dtype)
            
        bilinear_activation = self.bilinear(x) * x
        
        # Mix between the two based on alpha (but alpha=0, so it's just bilinear)
        return self.alpha * silu_activation + (1 - self.alpha) * bilinear_activation

class BilinearLoss(nn.Module):
    """
    Custom loss function that adds a regularization term based on interaction matrix Q
    
    The regularization term minimizes the Frobenius norm of Q, where Q is an interaction
    matrix between W, V and output vector u.
    
    We ensure the regularization term is weighted less than the next token prediction 
    loss, specifically targeting a ratio where reg_loss is ~1/3 of original_loss.
    """
    def __init__(self, original_loss_fn, transformer_blocks, u_vectors, lambda_reg=0.01):
        super().__init__()
        self.original_loss_fn = original_loss_fn
        self.transformer_blocks = transformer_blocks
        self.u_vectors = u_vectors  # Dictionary mapping block indices to their u_vectors
        self.lambda_reg = lambda_reg  # Regularization strength
        self.loss_stats = {'original': 0.0, 'regularization': 0.0, 'total': 0.0, 'ratio': 0.0}
        
    def compute_frobenius_norm_squared(self):
        """
        Compute the squared Frobenius norm of the interaction matrix Q 
        without explicitly constructing the full matrix in memory.
        
        This follows the formula from the provided equation:
        Q[a] = u^T * W_a * V_a^T
        
        We compute ||Q||_F^2 = sum(Q[a]^2) for all a
        """
        # Initialize running sum for Frobenius norm squared
        frob_norm_squared = 0.0
        
        for block_idx in self.transformer_blocks:
            try:
                # Extract W and V matrices from the transformer block
                block = self.model.model.layers[block_idx]
                
                # W matrix from gate_proj (maps from hidden_dim to intermediate_dim)
                W = block.mlp.gate_proj.weight  # shape: [intermediate_dim, hidden_dim]
                
                # V matrix from up_proj (maps from hidden_dim to intermediate_dim) 
                V = block.mlp.up_proj.weight  # shape: [intermediate_dim, hidden_dim]
                
                # Get the u_vector for this specific block
                u_vector = self.u_vectors[block_idx]
                
                # Debug info about shapes and dtypes
                logger.info(f"Block {block_idx} - W shape: {W.shape}, dtype: {W.dtype}")
                logger.info(f"Block {block_idx} - V shape: {V.shape}, dtype: {V.dtype}")
                logger.info(f"Block {block_idx} - u_vector shape: {u_vector.shape}, dtype: {u_vector.dtype}")
                
                # Ensure u_vector has the same dtype as W and V
                u_vector = u_vector.to(device=W.device, dtype=W.dtype)
                
                # Extra safety check for dtype
                assert W.dtype == u_vector.dtype, f"dtype mismatch: W {W.dtype} vs u_vector {u_vector.dtype}"
                assert V.dtype == u_vector.dtype, f"dtype mismatch: V {V.dtype} vs u_vector {u_vector.dtype}"
                
                # For each row 'a' in W and V, compute contribution to Frobenius norm
                for a in range(W.shape[0]):  # iterate over intermediate_dim
                    w_a = W[a]  # [hidden_dim]
                    v_a = V[a]  # [hidden_dim]
                    
                    # Calculate u^T * w_a (scalar)
                    u_dot_w = torch.matmul(u_vector, w_a)
                    
                    # Calculate q_a = u_dot_w * v_a
                    q_a = u_dot_w * v_a
                    
                    # Add ||q_a||^2 to the running sum
                    # (squared L2 norm of this vector)
                    frob_norm_squared += torch.sum(q_a ** 2)
                    
            except Exception as e:
                logger.error(f"Error in compute_frobenius_norm_squared for block {block_idx}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        return frob_norm_squared
    
    def forward(self, model, outputs, targets):
        """
        Compute the combined loss with the regularization term
        
        Args:
            model: The model being fine-tuned
            outputs: Output logits from the model
            targets: Target token IDs
            
        Returns:
            total_loss: Combined loss with regularization
        """
        try:
            # Store model reference to access layers during Q computation
            self.model = model
            
            # Compute original loss (next token prediction)
            original_loss = self.original_loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Log dtype of original loss
            logger.info(f"Original loss dtype: {original_loss.dtype}")
            
            # Compute Frobenius norm squared directly without storing full Q matrix
            Q_frob_norm_squared = self.compute_frobenius_norm_squared()
            
            # Log dtype of Q_frob_norm_squared
            logger.info(f"Q_frob_norm_squared dtype: {Q_frob_norm_squared.dtype}")
            
            # Make sure Q_frob_norm_squared is the same dtype as original_loss
            Q_frob_norm_squared = Q_frob_norm_squared.to(dtype=original_loss.dtype)
            
            # Verify conversion worked
            logger.info(f"After conversion, Q_frob_norm_squared dtype: {Q_frob_norm_squared.dtype}")
            
            # Apply regularization
            reg_loss = self.lambda_reg * Q_frob_norm_squared
            
            # Combine losses
            total_loss = original_loss + reg_loss
            
            # Store loss components for logging
            current_ratio = reg_loss.item() / original_loss.item() if original_loss.item() > 0 else float('inf')
            self.loss_stats['original'] = original_loss.item()
            self.loss_stats['regularization'] = reg_loss.item()
            self.loss_stats['total'] = total_loss.item()
            self.loss_stats['ratio'] = current_ratio
            
            # Dynamically adjust lambda_reg to maintain the desired ratio
            # We want reg_loss to be approximately 1/3 of original_loss
            if hasattr(self, 'auto_adjust_lambda') and self.auto_adjust_lambda:
                target_ratio = 1/3  # Target ratio of reg_loss/original_loss
                
                # Only adjust if the ratio is significantly off
                if current_ratio < target_ratio * 0.8 or current_ratio > target_ratio * 1.2:
                    # Adjust lambda to move closer to target ratio
                    adjustment_factor = target_ratio / current_ratio
                    self.lambda_reg *= adjustment_factor
                    logger.info(f"Adjusted lambda_reg to {self.lambda_reg:.6f} (ratio: {current_ratio:.4f}, target: {target_ratio:.4f})")
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error in BilinearLoss.forward: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

def check_bilinear_activations(model, transformer_indices):
    """
    Check if the specified transformer blocks already have InterpolatedSiLU activations.
    If not, log a warning as the model should already be in bilinear form.
    
    Args:
        model: The language model
        transformer_indices: List of transformer block indices to check
        
    Returns:
        model: The model (unmodified, as we expect it to already have bilinear activations)
    """
    bilinear_count = 0
    need_replacement = 0
    
    # Get model's dtype
    model_dtype = next(model.parameters()).dtype
    logger.info(f"Model dtype: {model_dtype}")
    
    for idx in transformer_indices:
        try:
            layer = model.model.layers[idx]
            
            # Check if the layer already has an InterpolatedSiLU activation
            if hasattr(layer.mlp, 'act_fn') and isinstance(layer.mlp.act_fn, InterpolatedSiLU):
                # Ensure alpha is set to 0 for full bilinear mode
                layer.mlp.act_fn.alpha = 0.0
                
                # Ensure the bilinear weight has the correct dtype
                if layer.mlp.act_fn.bilinear.weight.dtype != model_dtype:
                    logger.warning(f"Layer {idx} has bilinear weights with incorrect dtype: "
                                 f"{layer.mlp.act_fn.bilinear.weight.dtype}, converting to {model_dtype}")
                    layer.mlp.act_fn.bilinear.weight.data = layer.mlp.act_fn.bilinear.weight.data.to(dtype=model_dtype)
                
                bilinear_count += 1
                logger.info(f"Layer {idx} already has InterpolatedSiLU with alpha set to 0.0")
                
            elif hasattr(layer.mlp, 'act_fn') and isinstance(layer.mlp.act_fn, nn.SiLU):
                # If it's still a regular SiLU, we need to warn about this
                need_replacement += 1
                logger.warning(f"Layer {idx} has standard SiLU activation, expected InterpolatedSiLU")
                
                # Since the model should already be bilinear, we'll create a replacement
                # to ensure consistency
                hidden_size = layer.mlp.gate_proj.weight.shape[1]  # hidden_size is the input dimension of gate_proj
                logger.info(f"Creating InterpolatedSiLU with hidden_size={hidden_size} for layer {idx}")
                
                interpolated_activation = InterpolatedSiLU(hidden_size)
                
                # Ensure the new module is on the correct device and dtype
                device = layer.mlp.act_fn.device if hasattr(layer.mlp.act_fn, 'device') else layer.mlp.gate_proj.weight.device
                interpolated_activation = interpolated_activation.to(device=device, dtype=model_dtype)
                
                # Set alpha to 0.0 for full bilinear mode
                interpolated_activation.alpha = 0.0
                
                # Verify dtype
                logger.info(f"New InterpolatedSiLU bilinear weight dtype: {interpolated_activation.bilinear.weight.dtype}")
                
                # Replace the activation
                layer.mlp.act_fn = interpolated_activation
                logger.info(f"Replaced SiLU in layer {idx} with InterpolatedSiLU (alpha=0.0)")
                
            else:
                logger.warning(f"Layer {idx} has unknown activation type: {type(layer.mlp.act_fn)}")
                
        except Exception as e:
            logger.warning(f"Failed to check layer {idx}: {str(e)}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
    
    if bilinear_count > 0:
        logger.info(f"Found {bilinear_count} existing InterpolatedSiLU activations")
    
    if need_replacement > 0:
        logger.warning(
            f"Had to replace {need_replacement} standard SiLU activations with InterpolatedSiLU. "
            f"Please verify that the model checkpoint was actually in bilinear form."
        )
    
    return model

def set_bilinear_mode(model, transformer_indices, alpha=0.0):
    """
    Set the alpha parameter in InterpolatedSiLU modules to control the 
    interpolation between SiLU and bilinear modes.
    
    Args:
        model: The language model
        transformer_indices: List of transformer block indices to modify
        alpha: Interpolation parameter (0.0 = full bilinear, 1.0 = pure SiLU)
        
    Returns:
        model: Modified model
    """
    for idx in transformer_indices:
        try:
            layer = model.model.layers[idx]
            
            # Set alpha in the InterpolatedSiLU
            if hasattr(layer.mlp, 'act_fn') and isinstance(layer.mlp.act_fn, InterpolatedSiLU):
                layer.mlp.act_fn.alpha = alpha
                logger.info(f"Set alpha={alpha} in layer {idx}")
        except Exception as e:
            logger.warning(f"Failed to set alpha in layer {idx}: {str(e)}")
    
    return model

def save_bilinear_model(model, tokenizer, transformer_indices, u_vectors, save_dir, step=None):
    """
    Save the bilinear model, tokenizer, and u_vectors with additional metadata.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        transformer_indices: List of transformer block indices
        u_vectors: Dictionary mapping block indices to their linear probe weights
        save_dir: Base directory to save the model
        step: Current training step (for checkpoint naming)
        
    Returns:
        save_path: Path where the model was saved
    """
    # Create a subdirectory based on step if provided
    if step is not None:
        save_path = os.path.join(save_dir, f"checkpoint-{step}")
    else:
        save_path = os.path.join(save_dir, "final-model")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save additional metadata about the bilinear model
    bilinear_info = []
    for name, module in model.named_modules():
        if isinstance(module, InterpolatedSiLU):
            bilinear_info.append({
                "module_path": name,
                "alpha": float(module.alpha),
                "hidden_size": module.bilinear.weight.shape[0]
            })
    
    # Save bilinear configuration
    with open(os.path.join(save_path, "bilinear_config.json"), "w") as f:
        import json
        json.dump({
            "bilinear_modules": bilinear_info,
            "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Model with bilinear MLPs"
        }, f, indent=2)
    
    # Save the u_vectors for each transformer block
    u_vectors_save_path = os.path.join(save_path, "u_vectors")
    os.makedirs(u_vectors_save_path, exist_ok=True)
    
    for block_idx, u_vector in u_vectors.items():
        # Move to CPU before saving
        u_vector_cpu = u_vector.detach().cpu()
        torch.save(u_vector_cpu, os.path.join(u_vectors_save_path, f"u_vector_block_{block_idx}.pt"))
    
    # Save mapping info
    with open(os.path.join(u_vectors_save_path, "u_vectors_info.json"), "w") as f:
        import json
        json.dump({
            "transformer_blocks": transformer_indices,
            "u_vectors_files": [f"u_vector_block_{idx}.pt" for idx in transformer_indices],
            "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    logger.info(f"Saved bilinear model checkpoint to {save_path}")
    logger.info(f"Saved u_vectors for {len(u_vectors)} blocks to {u_vectors_save_path}")
    return save_path

def ensure_bilinear_weights_dtype(model):
    """
    Ensures that all InterpolatedSiLU modules in the model have their weights
    with the correct dtype (matching the model's dtype).
    
    This is critical to avoid dtype mismatches during forward passes.
    
    Args:
        model: The language model
        
    Returns:
        model: The model with consistent dtypes
    """
    model_dtype = next(model.parameters()).dtype
    logger.info(f"Ensuring all bilinear weights have dtype: {model_dtype}")
    
    count = 0
    converted = 0
    
    for name, module in model.named_modules():
        if isinstance(module, InterpolatedSiLU):
            count += 1
            current_dtype = module.bilinear.weight.dtype
            
            if current_dtype != model_dtype:
                logger.warning(f"Converting bilinear weights in {name} from {current_dtype} to {model_dtype}")
                module.bilinear.weight.data = module.bilinear.weight.data.to(dtype=model_dtype)
                converted += 1
    
    logger.info(f"Checked {count} InterpolatedSiLU modules, converted {converted} to {model_dtype}")
    return model

def get_u_vectors(model, transformer_indices, probe_weights_paths=None, output_dim=None):
    """
    Get the linear probe weights (u vector) for each transformer block.
    Either load from files or create random ones if not provided.
    
    Args:
        model: The model being fine-tuned
        transformer_indices: List of transformer block indices
        probe_weights_paths: List of paths to saved weights for each transformer block, or None
        output_dim: Dimensionality of the output space
        
    Returns:
        u_vectors: Dictionary mapping transformer block indices to their linear probe weights
    """
    # Get hidden dimension from model
    hidden_dim = model.config.hidden_size
    
    # Get model dtype and device for consistency
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    
    logger.info(f"Model dtype: {model_dtype}, device: {model_device}")
    
    # Initialize dictionary to store u_vectors for each transformer block
    u_vectors = {}
    
    # Process each transformer block
    for i, block_idx in enumerate(transformer_indices):
        # Check if we have a path for this specific block
        specific_path = None
        if probe_weights_paths and i < len(probe_weights_paths) and probe_weights_paths[i]:
            specific_path = probe_weights_paths[i]
        
        if specific_path and os.path.exists(specific_path):
            logger.info(f"Loading linear probe weights for block {block_idx} from {specific_path}")
            
            # Load the saved weights
            loaded_data = torch.load(specific_path, map_location="cpu")
            
            # Handle different types of saved data
            if isinstance(loaded_data, dict):
                # If it's a dictionary (state_dict), try to find the appropriate weights
                logger.info(f"Loaded dict with keys: {list(loaded_data.keys())}")
                
                # Look for common keys that might contain the probe weights
                if 'weight' in loaded_data:
                    u_vector = loaded_data['weight']
                    logger.info(f"Using weight tensor, shape: {u_vector.shape}, dtype: {u_vector.dtype}")
                elif 'probe' in loaded_data:
                    u_vector = loaded_data['probe']
                    logger.info(f"Using probe tensor, shape: {u_vector.shape}, dtype: {u_vector.dtype}")
                elif 'linear.weight' in loaded_data:
                    u_vector = loaded_data['linear.weight']
                    logger.info(f"Using linear.weight tensor, shape: {u_vector.shape}, dtype: {u_vector.dtype}")
                elif 'model.weight' in loaded_data:
                    u_vector = loaded_data['model.weight']
                    logger.info(f"Using model.weight tensor, shape: {u_vector.shape}, dtype: {u_vector.dtype}")
                elif len(loaded_data) == 1:
                    # If there's only one item, use that
                    key = list(loaded_data.keys())[0]
                    u_vector = loaded_data[key]
                    logger.info(f"Using single tensor with key '{key}', shape: {u_vector.shape}, dtype: {u_vector.dtype}")
                else:
                    # Log available keys and try to use the first tensor
                    logger.warning(f"Multiple keys in probe file, available keys: {list(loaded_data.keys())}")
                    for key, value in loaded_data.items():
                        if isinstance(value, torch.Tensor):
                            u_vector = value
                            logger.info(f"Using tensor from key '{key}', shape: {u_vector.shape}, dtype: {u_vector.dtype}")
                            break
                    else:
                        # If no tensor is found, fallback to random initialization
                        logger.warning(f"No suitable tensor found in {specific_path}, using random initialization")
                        u_vector = create_random_probe(hidden_dim, output_dim)
            elif isinstance(loaded_data, torch.Tensor):
                # If it's directly a tensor, use it
                u_vector = loaded_data
                logger.info(f"Loaded tensor directly, shape: {u_vector.shape}, dtype: {u_vector.dtype}")
            else:
                # Fallback to random if we can't interpret the loaded data
                logger.warning(f"Unexpected data type in {specific_path}: {type(loaded_data)}, using random initialization")
                u_vector = create_random_probe(hidden_dim, output_dim)
        else:
            logger.info(f"Creating random linear probe weights for block {block_idx}")
            u_vector = create_random_probe(hidden_dim, output_dim)
        
        # Ensure u_vector is converted to the model's dtype and device
        original_dtype = u_vector.dtype
        u_vector = u_vector.to(device=model_device, dtype=model_dtype)
        logger.info(f"Converted u_vector from {original_dtype} to {u_vector.dtype} for block {block_idx}")
        
        # Store u_vector for this block
        u_vectors[block_idx] = u_vector
    
    return u_vectors


def create_random_probe(hidden_dim, output_dim=None):
    """Helper function to create a random probe with proper normalization"""
    # If output_dim is not provided, default to a single dimension (binary classification)
    if output_dim is None:
        output_dim = 1
    
    # Create a random vector normalized to unit length
    if output_dim == 1:
        # For single output dimension case
        u_vector = torch.randn(hidden_dim)
        u_vector = u_vector / torch.norm(u_vector)
    else:
        # For multi-output case, create a matrix of shape [output_dim, hidden_dim]
        u_vector = torch.randn(output_dim, hidden_dim)
        # Normalize each row
        for i in range(output_dim):
            u_vector[i] = u_vector[i] / torch.norm(u_vector[i])
    
    return u_vector

def train_bilinear_model(args):
    """
    Main training function for bilinear model fine-tuning.
    
    Args:
        args: Command-line arguments
    """
    # Add traceback module for better error logging
    import traceback
    
    # Initialize wandb
    run_name = f"bilinear-finetuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project="bilinear-tinyllama",
        name=run_name,
        config=vars(args),
        notes="Finetuning TinyLlama with bilinear MLPs and Q matrix regularization"
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Log training configuration
    logger.info(f"Starting bilinear fine-tuning with configuration:")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"  {arg_name}: {arg_value}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
            trust_remote_code=True
        )
        
        # Log model's dtype
        logger.info(f"Model loaded with dtype: {next(model.parameters()).dtype}")
        
        # Parse transformer indices
        transformer_indices = [int(idx) for idx in args.transformer_indices.split(",")]
        logger.info(f"Target transformer blocks: {transformer_indices}")
        
        # Parse probe weights paths if provided
        probe_weights_paths = None
        if hasattr(args, 'probe_weights_paths') and args.probe_weights_paths:
            probe_weights_paths = [path.strip() for path in args.probe_weights_paths.split(",")]
            # Replace empty strings with None
            probe_weights_paths = [path if path else None for path in probe_weights_paths]
            
            # Log the mapping between transformer blocks and probe weights
            logger.info("Probe weights mapping:")
            for i, block_idx in enumerate(transformer_indices):
                path = probe_weights_paths[i] if i < len(probe_weights_paths) else None
                logger.info(f"  Block {block_idx}: {path if path else 'random initialization'}")
        
        # Check if the specified transformer blocks already have InterpolatedSiLU activations
        logger.info(f"Checking bilinear transformer blocks: {transformer_indices}")
        model = check_bilinear_activations(model, transformer_indices)
        
        # Move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model moved to {device}")

        model = ensure_bilinear_weights_dtype(model)
        
        # Freeze all parameters except for the specified transformer W and V matrices
        trainable_params, frozen_params = freeze_all_except_target_mlp_matrices(model, transformer_indices)
        
        # Log model information to wandb
        wandb.config.update({
            "model_name": args.model_path,
            "target_blocks": transformer_indices,
            "device": str(device),
            "precision": "bfloat16" if args.use_bf16 else "float16",
            "hidden_size": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "num_layers": model.config.num_hidden_layers
        })
        
        # Log GPU information
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                "cuda_version": torch.version.cuda
            }
            wandb.config.update(gpu_info)
            logger.info(f"Using GPU: {gpu_info['gpu_name']} with {gpu_info['gpu_memory_total']:.2f} GB memory")
        
        # Load or initialize the linear probe weights (u_vectors) for each transformer block
        u_vectors = get_u_vectors(model, transformer_indices, probe_weights_paths, args.output_dim)
        
        # Move u_vectors to the correct device and dtype
        for block_idx in u_vectors:
            original_dtype = u_vectors[block_idx].dtype
            u_vectors[block_idx] = u_vectors[block_idx].to(device=device, dtype=model.dtype)
            logger.info(f"Moved u_vector for block {block_idx} from {original_dtype} to {u_vectors[block_idx].dtype}")
        
        # Load dataset
        logger.info("Loading FineWeb dataset")
        raw_dataset = load_dataset(
            "HuggingFaceFW/fineweb", 
            split="train", 
            streaming=True
        )
        
        # Create training dataset
        train_dataset = TokenStreamingDataset(
            raw_dataset,
            tokenizer,
            max_length=args.max_length,
            target_tokens=args.total_tokens
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator
        )
        
        # Prepare optimizer with weight decay differentiation
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Double-check that we're only optimizing the right parameters
        trainable_param_count = sum(p.numel() for param_group in optimizer_grouped_parameters for p in param_group["params"])
        logger.info(f"Optimizer will train {trainable_param_count:,} parameters")
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon
        )
        
        # Learning rate scheduler
        num_training_steps = args.max_steps
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Cross-entropy loss for language modeling
        original_loss_fn = nn.CrossEntropyLoss()
        
        # Create custom loss with regularization
        bilinear_loss_fn = BilinearLoss(
            original_loss_fn=original_loss_fn,
            transformer_blocks=transformer_indices,
            u_vectors=u_vectors,
            lambda_reg=args.lambda_reg
        )
        
        # Add auto-adjustment for lambda if requested
        if args.auto_adjust_lambda:
            bilinear_loss_fn.auto_adjust_lambda = True
        
        # Set up training variables
        step = 0
        epochs = 0
        actual_train_tokens = 0
        accumulated_loss = 0
        gradient_accumulation_steps = args.grad_accum
        
        # Training loop
        model.train()
        training_start_time = datetime.now()
        
        # Main training loop
        while step < args.max_steps:
            logger.info(f"Starting epoch {epochs+1}")
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epochs+1}")
            epoch_start_time = datetime.now()
            
            for batch in epoch_iterator:
                try:
                    # Get input IDs and target IDs (shifted for next token prediction)
                    input_ids = batch["input_ids"].to(device)
                    logger.info(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
                    
                    # Shift input/target for language modeling
                    target_ids = input_ids.clone()
                    
                    # Forward pass
                    outputs = model(input_ids)
                    logger.info(f"Model output logits shape: {outputs.logits.shape}, dtype: {outputs.logits.dtype}")
                    
                    # Calculate loss with regularization term
                    loss = bilinear_loss_fn(model, outputs.logits, target_ids)
                    logger.info(f"Loss: {loss.item()}, dtype: {loss.dtype}")
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    
                    # Accumulate loss for logging
                    accumulated_loss += loss.item() * gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    logger.info("Backward pass completed")
                    
                    # Update weights after accumulating gradients
                    if (step + 1) % gradient_accumulation_steps == 0 or step == args.max_steps - 1:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        
                        # Check if any gradient is NaN
                        has_nan_gradient = False
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                if torch.isnan(param.grad).any():
                                    has_nan_gradient = True
                                    logger.warning(f"NaN gradient detected in {name} at step {step}")
                                    break
                        
                        if has_nan_gradient:
                            # Log NaN event to wandb
                            wandb.log({"nan_gradient_detected": 1}, step=step//gradient_accumulation_steps)
                        
                        # Update parameters
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        
                        # Log basic metrics
                        optimization_step = (step + 1) // gradient_accumulation_steps
                        
                        # Log to console
                        if optimization_step % args.logging_steps == 0:
                            logger.info(
                                f"Step {optimization_step}: "
                                f"loss={bilinear_loss_fn.loss_stats['total']:.4f} "
                                f"(original={bilinear_loss_fn.loss_stats['original']:.4f}, "
                                f"reg={bilinear_loss_fn.loss_stats['regularization']:.4f}, "
                                f"ratio={bilinear_loss_fn.loss_stats['ratio']:.4f}), "
                                f"lr={lr_scheduler.get_last_lr()[0]:.8f}, "
                                f"tokens={actual_train_tokens:,}"
                            )
                            accumulated_loss = 0
                    
                    # Track tokens processed in this batch
                    tokens_in_batch = input_ids.numel()
                    actual_train_tokens += tokens_in_batch
                    
                    # Update progress bar
                    epoch_iterator.set_postfix(
                        loss=bilinear_loss_fn.loss_stats['total'],
                        reg_ratio=bilinear_loss_fn.loss_stats['ratio'],
                        lr=lr_scheduler.get_last_lr()[0]
                    )
                    
                    # Increment step
                    step += 1
                    
                except Exception as e:
                    logger.error(f"Error during training step {step}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def freeze_all_except_target_mlp_matrices(model, transformer_indices):
    """
    Freeze all model parameters except the W (gate_proj) and V (up_proj) matrices
    in the specified transformer blocks.
    
    Args:
        model: The language model
        transformer_indices: List of transformer block indices where W and V should remain trainable
        
    Returns:
        trainable_params: List of trainable parameter names (for logging)
        frozen_params: List of frozen parameter names (for logging)
    """
    # First, freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then, unfreeze only the target parameters
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        is_trainable = False
        
        # Check if this parameter is a W or V matrix in one of our target blocks
        for block_idx in transformer_indices:
            # Match gate_proj (W) and up_proj (V) in specified layers
            if (f"layers.{block_idx}.mlp.gate_proj.weight" in name or 
                f"layers.{block_idx}.mlp.up_proj.weight" in name):
                param.requires_grad = True
                is_trainable = True
                trainable_params.append(name)
                break
        
        if not is_trainable:
            frozen_params.append(name)
    
    # Verify that we're only training the right parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_param_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_param_count:,} ({len(trainable_params)} parameter groups)")
    logger.info(f"Frozen parameters: {frozen_param_count:,} ({len(frozen_params)} parameter groups)")
    
    # For sanity check, log the trainable parameter names
    logger.info(f"Trainable parameters: {trainable_params}")
    
    # Make sure everything adds up
    assert total_params == trainable_param_count + frozen_param_count, "Parameter count mismatch!"
    
    return trainable_params, frozen_params



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune a TinyLlama model with bilinear MLPs")
    
    # Model and data parameters
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to pretrained bilinear model")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--transformer_indices", type=str, required=True, 
                        help="Comma-separated list of transformer block indices to modify")
    parser.add_argument("--probe_weights_paths", type=str, default=None, 
                        help="Comma-separated list of paths to linear probe weights, one for each transformer index")
    parser.add_argument("--output_dim", type=int, default=1, 
                        help="Dimension of the output space for the linear probe")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--grad_accum", type=int, default=8, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Initial learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, 
                        help="Maximum number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=100, 
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length")
    parser.add_argument("--lambda_reg", type=float, default=0.01, 
                        help="Regularization strength for Q matrix")
    parser.add_argument("--auto_adjust_lambda", action="store_true", 
                        help="Automatically adjust lambda to maintain desired ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, 
                        help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, 
                        help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, 
                        help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial"], 
                        help="Learning rate scheduler type")
    
    # Data parameters
    parser.add_argument("--total_tokens", type=int, default=100_000_000, 
                        help="Total number of tokens to train on")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=10, 
                        help="Log training info every X steps")
    parser.add_argument("--save_steps", type=int, default=0, 
                        help="Save checkpoint every X steps (0 to disable step-based saving)")
    
    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, default="bilinear-tinyllama", 
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, 
                        help="Weights & Biases entity (team) name")
    parser.add_argument("--wandb_name", type=str, default=None, 
                        help="Weights & Biases run name")
    parser.add_argument("--wandb_tags", type=str, default=None, 
                        help="Comma-separated tags for Weights & Biases run")
    parser.add_argument("--save_to_wandb", action="store_true", 
                        help="Save model checkpoints to W&B")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--use_bf16", action="store_true", 
                        help="Use bfloat16 precision instead of float16")
    
    args = parser.parse_args()
    
    # Process wandb tags if provided
    if args.wandb_tags:
        wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",")]
    else:
        wandb_tags = None
    
    # Override wandb project and entity if provided
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_name:
        os.environ["WANDB_NAME"] = args.wandb_name
    
    # Parse probe weights paths
    probe_weights_paths = None
    if args.probe_weights_paths:
        probe_weights_paths = [path.strip() for path in args.probe_weights_paths.split(",")]
        # Replace empty strings with None
        probe_weights_paths = [path if path else None for path in probe_weights_paths]
    
    # Run the training
    train_bilinear_model(args)
