#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune TinyLlama by gradually removing nonlinearities (SwiGLU activations).
This script linearly scales the beta parameter from 1 to 0 over a specified portion
of training, effectively converting the SwiGLU activation to a linear function.
"""

import os
import sys
import types
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    set_seed,
    DataCollatorForLanguageModeling,
)
import datasets
from einops import einsum
import gc
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define decay functions for beta parameter
def linear_decay(x, decay_point=0.3):
    """
    Linear decay function that scales beta from 1 to 0 during the first decay_point portion
    of training and stays at 0 for the remaining portion.
    
    Args:
        x: fraction of training complete (0 to 1)
        decay_point: fraction of training at which beta reaches 0 (default: 0.3)
    
    Returns:
        beta value
    """
    if x < decay_point:
        # Linear interpolation from 1 to 0 during first decay_point portion
        return 1.0 - (x / decay_point)
    else:
        # Stay at 0 for the remaining portion
        return 0.0

def exp_decay(x, k=1.05):
    """Exponential decay function for beta parameter"""
    return min(x * np.e ** (k - k * x), 1.0)

def logistic_decay(x, d=4):
    """Logistic decay function for beta parameter"""
    return 1 / (1 + np.exp(-2*d*x + d))

def smootherstep_decay(x, s=0.5):
    """Smootherstep decay function for beta parameter"""
    return (6*x**5 - 15*x**4 + 10*x**3)**s

def no_decay(x):
    """No decay - keep beta constant at a small value"""
    return 1e-5

# Custom activation class that implements SwiGLU with modifiable beta
class SwiGLUWithBeta(torch.nn.Module):
    """Swish-beta activation function with modifiable beta parameter"""
    def __init__(self):
        super().__init__()
        self.beta = 1.0  # Start with original beta=1 (SiLU)
        
    def forward(self, x, gate=None):
        """
        If gate is None, applies Swish-beta to x: x * sigmoid(beta*x)
        If gate is provided, applies SwiGLU: x * (gate * sigmoid(beta*gate))
        """
        if gate is None:
            return x * torch.sigmoid(self.beta * x)
        else:
            return x * (gate * torch.sigmoid(self.beta * gate))

def replace_swiglu_with_interpolators(model):
    """
    Replace SwiGLU activation functions with beta-controllable versions
    that will modulate the beta parameter during training.
    
    This function handles different model architectures including TinyLlama.
    
    Args:
        model: The pre-trained language model
        
    Returns:
        The model with modifiable SwiGLU activations
    """
    logger.info(f"Model architecture: {type(model).__name__}")
    
    # Check which model we're working with and adapt accordingly
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Transformers library standard layout for LLaMA-style models
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # Older models or custom layouts
        layers = model.transformer.h
    else:
        raise ValueError(f"Unsupported model architecture: {type(model).__name__}")
    
    logger.info(f"Found {len(layers)} transformer layers to modify")
    
    # Create a single instance to be used model-wide
    swiglu_fn = SwiGLUWithBeta()
    
    # Track how many activations we modified
    modified_count = 0
    
    # Monkey patch the SwiGLU activation in each layer
    for i, layer in enumerate(layers):
        # Check for different MLP architectures
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            
            # Common architecture pattern in modern models
            if hasattr(mlp, 'act_fn') and callable(mlp.act_fn):
                # Replace activation function directly
                mlp._original_act_fn = mlp.act_fn
                mlp.act_fn = swiglu_fn
                modified_count += 1
            
            # Handle architectures with explicit gate and up projections (LLaMA style)
            elif hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj'):
                # Store original forward method
                original_forward = mlp.forward
                
                # Create new forward method with beta-controlled activation
                def new_forward(self, x):
                    gate = self.gate_proj(x)
                    up = self.up_proj(x)
                    # Use our beta-controlled activation
                    x = swiglu_fn(up, gate)
                    return self.down_proj(x)
                
                # Bind the new method to the MLP module
                mlp.forward = types.MethodType(new_forward, mlp)
                modified_count += 1
                
            # TinyLlama's specific architecture
            elif hasattr(mlp, 'w') and hasattr(mlp.w, 'gate'):
                mlp.w._original_gate = mlp.w.gate
                
                # Create a custom forward method for the gate
                class CustomGate(torch.nn.Module):
                    def __init__(self, original_gate):
                        super().__init__()
                        self.original_gate = original_gate
                        self.beta = 1.0
                    
                    def forward(self, x):
                        # Apply the gate but modulate the beta in the Swish function
                        result = x * torch.sigmoid(self.beta * x)
                        return result
                
                mlp.w.gate = CustomGate(mlp.w.gate)
                modified_count += 1
    
    logger.info(f"Modified {modified_count} activation functions with beta-controlled Swish")
    
    # Add a convenience method to update beta globally
    def update_beta(model, beta_value):
        # Update the global swiglu function
        swiglu_fn.beta = beta_value
        
        # Also update any specific gate instances
        for layer in layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'w') and hasattr(layer.mlp.w, 'gate'):
                if hasattr(layer.mlp.w.gate, 'beta'):
                    layer.mlp.w.gate.beta = beta_value
    
    # Attach this method to the model for easy access
    model.update_beta = types.MethodType(update_beta, model)
    
    return model

class BetaDecayCallback(TrainerCallback):
    """
    Callback to control beta parameter decay during training.
    Also logs metrics to Weights & Biases and generates visualizations.
    """
    def __init__(self, model, decay_func=linear_decay, decay_point=0.3):
        self.model = model
        self.decay_func = decay_func
        self.decay_point = decay_point
        # Track beta and loss values for plotting
        self.steps = []
        self.beta_values = []
        self.loss_values = []
 
    def on_step_begin(self, args, state, control, **kwargs):
        """Update beta value before each training step"""
        fraction = state.global_step / state.max_steps
        beta = self.decay_func(x=fraction, decay_point=self.decay_point)
        
        # Use the model's update_beta method if available
        if hasattr(self.model, 'update_beta'):
            self.model.update_beta(beta)
        else:
            # Fall back to updating individual modules
            for module in self.model.modules():
                if hasattr(module, 'beta'):
                    module.beta = beta
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log beta value and other metrics"""
        if logs is None:
            return
            
        fraction = state.global_step / state.max_steps
        beta = self.decay_func(x=fraction, decay_point=self.decay_point)
        logs['beta'] = beta
        
        # Log to W&B if initialized
        if wandb.run is not None:
            wandb.log({'beta': beta}, step=state.global_step)
        
        # Store values for plotting
        self.steps.append(state.global_step)
        self.beta_values.append(beta)
        if 'loss' in logs:
            self.loss_values.append(logs['loss'])
    
    def on_train_end(self, args, state, control, **kwargs):
        """Generate final plots at the end of training"""
        self.plot_training_progress(os.path.join(args.output_dir, "training_progress.png"))
        
        # Log final plot to W&B
        if wandb.run is not None:
            wandb.log({"training_progress": wandb.Image(
                os.path.join(args.output_dir, "training_progress.png"))
            })
    
    def plot_training_progress(self, save_path="training_progress.png"):
        """Generate and save a plot showing beta decay and loss during training"""
        if not self.steps:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot beta values
        ax1.plot(self.steps, self.beta_values, 'b-', label='Beta')
        ax1.set_ylabel('Beta value')
        ax1.set_title('Beta Decay During Training')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss if available
        if self.loss_values:
            ax2.plot(self.steps, self.loss_values, 'r-', label='Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Training steps')
            ax2.set_title('Training Loss')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune"""
    model_name_or_path: str = field(
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer implementations"}
    )
    revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (branch name, tag name or commit id)"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the remote code when loading the model from HF Hub"}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training"""
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use"}
    )
    dataset_path: Optional[str] = field(
        default="./fineweb_slice",
        metadata={"help": "Path to local dataset directory"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for training"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for preprocessing"}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Custom training arguments with additional parameters for beta decay"""
    beta_decay_function: str = field(
        default="linear",
        metadata={"help": "Beta decay function: linear, exp, logistic, smootherstep, or none"}
    )
    beta_decay_point: float = field(
        default=0.3,
        metadata={"help": "Percentage of training to decay beta from 1 to 0 (default: 30%)"}
    )

def main():
    """Main training function"""
    # Parse arguments using HfArgumentParser
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    
    # Try to parse args from command line, or from a config file
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass a single JSON file, parse that instead
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Otherwise parse command line args
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialize wandb for experiment tracking
    if training_args.report_to and "wandb" in training_args.report_to:
        wandb.init(
            project="tinyllama-nonlinearity-removal",
            config={
                "model": model_args.model_name_or_path,
                "beta_decay_function": training_args.beta_decay_function,
                "beta_decay_point": training_args.beta_decay_point,
                "max_seq_length": data_args.max_seq_length,
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
            }
        )
    
    # Set the beta decay function based on the argument
    decay_functions = {
        "linear": linear_decay,
        "exp": exp_decay,
        "logistic": logistic_decay,
        "smootherstep": smootherstep_decay,
        "none": no_decay
    }
    decay_func = decay_functions.get(training_args.beta_decay_function, linear_decay)
    
    # Set seeds for reproducibility
    if training_args.seed is not None:
        set_seed(training_args.seed)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Fix for LLaMA-based tokenizers that don't have a pad token set
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token since it was not set")
        tokenizer.pad_token = tokenizer.eos_token
        # If needed, you can also resize the embedding
        # special_tokens_dict = {'pad_token': '[PAD]'}
        # tokenizer.add_special_tokens(special_tokens_dict)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Resize model embeddings if we added new tokens to the tokenizer
    if tokenizer.pad_token != tokenizer.eos_token and model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        logger.info(f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Replace SwiGLU with beta-controllable version
    logger.info("Replacing SwiGLU activations with beta-controlled versions")
    model = replace_swiglu_with_interpolators(model)
    
    # Load dataset
    logger.info("Loading dataset")
    raw_dataset = None
    
    # Try multiple loading methods
    if data_args.dataset_name:
        try:
            # Load from Hugging Face Hub
            raw_dataset = datasets.load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split="train",
            )
            logger.info(f"Loaded {data_args.dataset_name} dataset from Hugging Face Hub")
        except Exception as e:
            logger.warning(f"Failed to load dataset from HF Hub: {e}")
    
    # If HF Hub failed or wasn't specified, try local loading
    if raw_dataset is None and os.path.exists(data_args.dataset_path):
        try:
            # Try loading as a directory of dataset files
            raw_dataset = datasets.load_from_disk(data_args.dataset_path)
            logger.info(f"Loaded dataset from {data_args.dataset_path}")
        except Exception as e:
            logger.warning(f"Failed to load dataset from disk: {e}")
            
            # Try loading as a CSV file
            if data_args.dataset_path.endswith(".csv"):
                try:
                    raw_dataset = datasets.load_dataset("csv", data_files=data_args.dataset_path)["train"]
                    logger.info(f"Loaded CSV dataset from {data_args.dataset_path}")
                except Exception as csv_e:
                    logger.warning(f"Failed to load CSV: {csv_e}")
            
            # Try loading as a JSON file
            elif data_args.dataset_path.endswith(".json"):
                try:
                    raw_dataset = datasets.load_dataset("json", data_files=data_args.dataset_path)["train"]
                    logger.info(f"Loaded JSON dataset from {data_args.dataset_path}")
                except Exception as json_e:
                    logger.warning(f"Failed to load JSON: {json_e}")
            
            # Try loading as a text file
            elif data_args.dataset_path.endswith(".txt"):
                try:
                    with open(data_args.dataset_path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    raw_dataset = datasets.Dataset.from_dict({"text": lines})
                    logger.info(f"Loaded text file from {data_args.dataset_path}")
                except Exception as txt_e:
                    logger.warning(f"Failed to load text file: {txt_e}")
    
    # If all loading methods failed, use a dummy dataset
    if raw_dataset is None:
        logger.warning("Using dummy dataset. Replace with actual data for real training.")
        dummy_texts = [
            "This is a dummy text for testing the fine-tuning process.",
            "Replace this with actual FineWeb data for real training."
        ] * 1000
        raw_dataset = datasets.Dataset.from_dict({"text": dummy_texts})
    
    # Calculate total tokens for reporting
    try:
        sample_size = min(100, len(raw_dataset))
        # Identify the text column
        text_column = None
        for col in raw_dataset.column_names:
            if col in ["text", "content"]:
                text_column = col
                break
        
        if text_column is None:
            # Try to find first string column
            for col in raw_dataset.column_names:
                if isinstance(raw_dataset[col][0], str):
                    text_column = col
                    break
        
        if text_column:
            sample_texts = [text for text in raw_dataset[text_column][:sample_size] if isinstance(text, str)]
            if sample_texts:
                avg_token_len = sum(len(tokenizer.encode(text)) for text in sample_texts) / len(sample_texts)
                total_tokens = avg_token_len * len(raw_dataset)
                logger.info(f"Estimated total tokens: {total_tokens/1e6:.2f}M")
            else:
                logger.warning("Could not estimate tokens: no valid text samples found")
        else:
            logger.warning("Could not estimate tokens: no text column identified")
    except Exception as e:
        logger.warning(f"Error calculating tokens: {e}")
    
    # Tokenize the dataset
    logger.info("Tokenizing dataset")
    def tokenize_function(examples):
        # Make sure we have text data
        texts = examples["text"] if "text" in examples else examples.get("content", [])
        
        if not texts:
            logger.warning(f"No text field found in examples: {list(examples.keys())}")
            # Try to get the first string field
            for key, value in examples.items():
                if isinstance(value, list) and all(isinstance(x, str) for x in value):
                    texts = value
                    logger.info(f"Using '{key}' as text field")
                    break
        
        # Apply tokenization
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
            return_tensors=None,  # Ensure we return lists not tensors
        )
        
        # For causal language modeling, we need to set the labels to be the same as input_ids
        # This is crucial for calculating the loss
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Identify text column to remove after tokenization
    if "text" in raw_dataset.column_names:
        text_column = "text"
    elif "content" in raw_dataset.column_names:
        text_column = "content"
    else:
        # Try to identify the text column by finding the first string column
        text_column = None
        for col in raw_dataset.column_names:
            if isinstance(raw_dataset[col][0], str):
                text_column = col
                logger.info(f"Using '{col}' as the text column")
                break
    
    # List of columns to remove (all except those needed for training)
    remove_columns = raw_dataset.column_names if text_column else []
    
    # Apply tokenization
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing dataset",
        num_proc=data_args.preprocessing_num_workers,
    )
    
    # Display sample data to verify dataset structure
    if len(tokenized_dataset) > 0:
        logger.info("Checking tokenized dataset sample:")
        sample = tokenized_dataset[0]
        for key, value in sample.items():
            # Truncate long values for display
            if isinstance(value, list) and len(value) > 10:
                logger.info(f"  {key}: {value[:10]}... (length {len(value)})")
            else:
                logger.info(f"  {key}: {value}")
        
        # Verify that we have the necessary keys for language modeling
        required_keys = ["input_ids", "attention_mask", "labels"]
        missing_keys = [key for key in required_keys if key not in sample]
        
        if missing_keys:
            logger.warning(f"Dataset missing required keys: {missing_keys}")
            
            # If labels are missing but we have input_ids, fix by adding labels
            if "labels" in missing_keys and "input_ids" in sample:
                logger.info("Adding labels to dataset (copying from input_ids)")
                def add_labels(example):
                    example["labels"] = example["input_ids"].copy()
                    return example
                
                tokenized_dataset = tokenized_dataset.map(
                    add_labels,
                    desc="Adding labels"
                )
    
    # Create the beta decay callback
    beta_callback = BetaDecayCallback(
        model=model, 
        decay_func=decay_func, 
        decay_point=training_args.beta_decay_point
    )
    
    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Add the data collator
        callbacks=[beta_callback],
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Generate and save the training progress plot
    logger.info("Generating training progress visualization")
    beta_callback.plot_training_progress(os.path.join(training_args.output_dir, "training_progress.png"))
    
    # Save the final model
    logger.info("Saving final model")
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    
    # Finish W&B run
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    # Make sure we have all required imports
    from transformers import HfArgumentParser, set_seed
    
    try:
        main()
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        
        # If wandb is running, log the error
        if wandb.run is not None:
            wandb.run.finish(exit_code=1)
            
        raise