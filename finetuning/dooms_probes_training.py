#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from torch import nn
from torch.optim import AdamW
import wandb
from tqdm import tqdm
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class HookManager:
    """
    Class to register and manage hooks into model layers.
    """
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.activations = {}
        self.layer_outputs = {}
        
    def register_hook(self, layer_names):
        """
        Register hooks for specified layers.
        
        Args:
            layer_names: List of layer names to register hooks for
        """
        # Identify the model architecture
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA/TinyLlama style models
            layers = self.model.model.layers
            for i, layer in enumerate(layers):
                layer_name = f"layer_{i}"
                if layer_name in layer_names:
                    # Register hook for MLP output only
                    layer.mlp.register_forward_hook(
                        lambda module, inp, out, layer_idx=i: 
                        self._hook_fn(f"mlp_out_{layer_idx}", out)
                    )
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT style models
            layers = self.model.transformer.h
            for i, layer in enumerate(layers):
                layer_name = f"layer_{i}"
                if layer_name in layer_names:
                    # Register hook for MLP output only
                    layer.mlp.register_forward_hook(
                        lambda module, inp, out, layer_idx=i: 
                        self._hook_fn(f"mlp_out_{layer_idx}", out)
                    )
        else:
            raise ValueError("Unsupported model architecture. Could not register hooks.")
            
        logger.info(f"Registered MLP output hooks for {len(layer_names)} layers")
    
    def _hook_fn(self, name, output):
        """
        Hook function that saves activations.
        """
        if isinstance(output, tuple):
            output = output[0]  # Get first item if it's a tuple
        self.activations[name] = output.detach()
        return None
    
    def get_activations(self):
        """
        Get the stored activations.
        """
        return self.activations
    
    def clear_activations(self):
        """
        Clear stored activations to free memory.
        """
        self.activations = {}


class LinearProbe(nn.Module):
    """
    Linear probe model for feature extraction.
    Always averages activations across all tokens.
    """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create a simple linear layer with bias
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Print dimensions during initialization
        print(f"Created LinearProbe with input_dim={input_dim}, output_dim={output_dim}")
        print(f"Linear layer weight shape: {self.linear.weight.shape}")
        
    def forward(self, x):
        # Convert input to float32 to match probe's dtype
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        
        # Handle different input shapes
        if len(x.shape) == 3:  # [batch, seq_len, hidden]
            # Average across all tokens in sequence
            x = torch.mean(x, dim=1)
            
        # Check dimensions
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[-1]}")
        
        # Apply linear layer and dropout
        return self.dropout(self.linear(x))
        
    def get_direction(self):
        """
        Returns the weight vector representing the direction in activation space
        that this probe is sensitive to.
        """
        return self.linear.weight.data.clone()


class ProbeTrainer:
    """
    Trainer for linear probes.
    """
    def __init__(
        self, 
        model, 
        tokenizer,
        layers_to_probe,
        feature_dim,
        learning_rate=1e-3,
        weight_decay=0.01,
        device=None,
        probe_hidden_dim=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers_to_probe = layers_to_probe
        self.feature_dim = feature_dim
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hook_manager = HookManager(model)
        self.hook_manager.register_hook(layers_to_probe)
        
        # Get hidden dimensions from model config
        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        elif hasattr(model.config, 'd_model'):
            self.hidden_dim = model.config.d_model
        else:
            logger.warning("Could not determine hidden dimension from model config. Using 2048.")
            self.hidden_dim = 2048
            
        self.probe_hidden_dim = probe_hidden_dim or self.hidden_dim
        
        # Make sure model and probes use compatible dtypes
        self.model_dtype = next(model.parameters()).dtype
        logger.info(f"Model is using dtype: {self.model_dtype}")
        logger.info(f"Hidden dimension: {self.hidden_dim}, Feature dimension: {self.feature_dim}")
        
        # Initialize probes for each layer
        self.probes = {}
        for layer in layers_to_probe:
            # We create only MLP output probes
            layer_idx = int(layer.split('_')[1])
            self.probes[f"mlp_out_{layer_idx}"] = LinearProbe(
                self.hidden_dim, self.feature_dim).to(self.device)
        
        # Set probe dtype to float32 for better training stability
        for name, probe in self.probes.items():
            # Move to device and ensure float32
            probe.to(self.device).to(torch.float32)
            
            # Log probe weight shapes
            for param_name, param in probe.named_parameters():
                if 'weight' in param_name:
                    logger.info(f"Probe {name} weight shape: {param.shape}, dtype: {param.dtype}")
        
        # Initialize optimizers
        self.optimizers = {}
        for name, probe in self.probes.items():
            self.optimizers[name] = AdamW(
                probe.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        logger.info(f"Initialized {len(self.probes)} probes for {len(layers_to_probe)} layers")
    
    def train(
        self, 
        train_dataloader, 
        val_dataloader=None, 
        epochs=3, 
        log_interval=10,
        use_wandb=False
    ):
        """
        Train the linear probes.
        """
        # Set model to eval mode since we're only training the probes
        self.model.eval()
        
        # Training metrics
        metrics = {name: {
            'train_loss': [], 
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_mse': [],
            'val_mse': [],
            'train_r2': [],
            'val_r2': []
        } for name in self.probes.keys()}
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            for probe_name in self.probes.keys():
                self.probes[probe_name].train()
                
            train_losses = {name: 0.0 for name in self.probes.keys()}
            train_accuracies = {name: [] for name in self.probes.keys()}
            train_mses = {name: [] for name in self.probes.keys()}
            train_r2s = {name: [] for name in self.probes.keys()}
            train_steps = 0
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    
                labels = batch['labels'].to(self.device)
                
                # Ensure labels have the right shape [batch_size, feature_dim]
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                
                # Forward pass through model to get activations
                with torch.no_grad():
                    _ = self.model(inputs, attention_mask=attention_mask)
                    activations = self.hook_manager.get_activations()
                
                # Train each probe
                for probe_name, probe in self.probes.items():
                    if probe_name in activations:
                        # Get activations for this probe
                        act = activations[probe_name]
                        
                        # Forward pass through probe (averaging happens in the probe)
                        self.optimizers[probe_name].zero_grad()
                        outputs = probe(act)
                        
                        # Calculate loss
                        loss = self.loss_fn(outputs, labels)
                        loss.backward()
                        
                        # Update weights
                        self.optimizers[probe_name].step()
                        
                        # Track metrics
                        train_losses[probe_name] += loss.item()
                        
                        # Calculate accuracy metrics (on CPU)
                        with torch.no_grad():
                            outputs_np = outputs.detach().cpu().numpy()
                            labels_np = labels.detach().cpu().numpy()
                            
                            # For binary classification (if applicable)
                            if self.feature_dim == 1:
                                # Binary accuracy (threshold at 0.5)
                                predicted_labels = (outputs_np > 0.5).astype(int)
                                true_labels = (labels_np > 0.5).astype(int)
                                accuracy = accuracy_score(true_labels, predicted_labels)
                                train_accuracies[probe_name].append(accuracy)
                            
                            # Regression metrics
                            mse = mean_squared_error(labels_np, outputs_np)
                            train_mses[probe_name].append(mse)
                            
                            # Try to calculate R², but handle edge cases
                            try:
                                r2 = r2_score(labels_np, outputs_np)
                                # R² can be negative; clip to prevent misleading results
                                r2 = max(-1.0, r2)  
                            except:
                                r2 = -1.0
                            train_r2s[probe_name].append(r2)
                
                train_steps += 1
                
                # Log training progress
                if batch_idx % log_interval == 0:
                    log_message = f"Batch {batch_idx}/{len(train_dataloader)}, "
                    for probe_name in list(self.probes.keys())[:3]:  # Show first 3 probes for brevity
                        log_message += f"{probe_name} loss: {train_losses[probe_name]/max(1, train_steps):.6f}, "
                        if train_accuracies[probe_name]:
                            log_message += f"acc: {np.mean(train_accuracies[probe_name]):.4f}, "
                        if train_r2s[probe_name]:
                            log_message += f"R²: {np.mean(train_r2s[probe_name]):.4f}, "
                    logger.info(log_message)
                
                # Clear activations to save memory
                self.hook_manager.clear_activations()
            
            # Calculate average training metrics
            for probe_name in self.probes.keys():
                avg_train_loss = train_losses[probe_name] / max(1, train_steps)
                metrics[probe_name]['train_loss'].append(avg_train_loss)
                
                # Average other metrics
                if train_accuracies[probe_name]:
                    avg_train_accuracy = np.mean(train_accuracies[probe_name])
                    metrics[probe_name]['train_accuracy'].append(float(avg_train_accuracy))
                
                if train_mses[probe_name]:
                    avg_train_mse = np.mean(train_mses[probe_name])
                    metrics[probe_name]['train_mse'].append(float(avg_train_mse))
                    
                if train_r2s[probe_name]:
                    avg_train_r2 = np.mean(train_r2s[probe_name])
                    metrics[probe_name]['train_r2'].append(float(avg_train_r2))
                
                if use_wandb:
                    wandb_logs = {
                        f"{probe_name}_train_loss": avg_train_loss,
                    }
                    if train_accuracies[probe_name]:
                        wandb_logs[f"{probe_name}_train_accuracy"] = avg_train_accuracy
                    if train_r2s[probe_name]:
                        wandb_logs[f"{probe_name}_train_r2"] = avg_train_r2
                    wandb.log(wandb_logs, step=epoch)
            
            # Validation
            if val_dataloader:
                for probe_name in self.probes.keys():
                    self.probes[probe_name].eval()
                
                val_losses = {name: 0.0 for name in self.probes.keys()}
                val_accuracies = {name: [] for name in self.probes.keys()}
                val_mses = {name: [] for name in self.probes.keys()}
                val_r2s = {name: [] for name in self.probes.keys()}
                val_steps = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc="Validation"):
                        inputs = batch['input_ids'].to(self.device)
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        # Ensure labels have the right shape [batch_size, feature_dim]
                        if labels.dim() == 1:
                            labels = labels.unsqueeze(1)
                        
                        # Forward pass through model to get activations
                        _ = self.model(inputs, attention_mask=attention_mask)
                        activations = self.hook_manager.get_activations()
                        
                        # Evaluate each probe
                        for probe_name, probe in self.probes.items():
                            if probe_name in activations:
                                # Get activations for this probe
                                act = activations[probe_name]
                                
                                # Forward pass through probe
                                outputs = probe(act)
                                
                                # Calculate loss
                                loss = self.loss_fn(outputs, labels)
                                
                                # Track metrics
                                val_losses[probe_name] += loss.item()
                                
                                # Calculate accuracy metrics (on CPU)
                                outputs_np = outputs.cpu().numpy()
                                labels_np = labels.cpu().numpy()
                                
                                # For binary classification (if applicable)
                                if self.feature_dim == 1:
                                    # Binary accuracy (threshold at 0.5)
                                    predicted_labels = (outputs_np > 0.5).astype(int)
                                    true_labels = (labels_np > 0.5).astype(int)
                                    accuracy = accuracy_score(true_labels, predicted_labels)
                                    val_accuracies[probe_name].append(accuracy)
                                
                                # Regression metrics
                                mse = mean_squared_error(labels_np, outputs_np)
                                val_mses[probe_name].append(mse)
                                
                                # Try to calculate R², but handle edge cases
                                try:
                                    r2 = r2_score(labels_np, outputs_np)
                                    # R² can be negative; clip to prevent misleading results
                                    r2 = max(-1.0, r2)
                                except:
                                    r2 = -1.0
                                val_r2s[probe_name].append(r2)
                        
                        val_steps += 1
                        
                        # Clear activations to save memory
                        self.hook_manager.clear_activations()
                
                # Calculate average validation metrics
                for probe_name in self.probes.keys():
                    avg_val_loss = val_losses[probe_name] / max(1, val_steps)
                    metrics[probe_name]['val_loss'].append(avg_val_loss)
                    
                    # Log validation metrics
                    log_message = f"Epoch {epoch+1}, {probe_name} val_loss: {avg_val_loss:.6f}"
                    
                    # Average other metrics
                    if val_accuracies[probe_name]:
                        avg_val_accuracy = np.mean(val_accuracies[probe_name])
                        metrics[probe_name]['val_accuracy'].append(float(avg_val_accuracy))
                        log_message += f", val_accuracy: {avg_val_accuracy:.4f}"
                    
                    if val_mses[probe_name]:
                        avg_val_mse = np.mean(val_mses[probe_name])
                        metrics[probe_name]['val_mse'].append(float(avg_val_mse))
                        
                    if val_r2s[probe_name]:
                        avg_val_r2 = np.mean(val_r2s[probe_name])
                        metrics[probe_name]['val_r2'].append(float(avg_val_r2))
                        log_message += f", val_R²: {avg_val_r2:.4f}"
                    
                    logger.info(log_message)
                    
                    if use_wandb:
                        wandb_logs = {
                            f"{probe_name}_val_loss": avg_val_loss,
                        }
                        if val_accuracies[probe_name]:
                            wandb_logs[f"{probe_name}_val_accuracy"] = avg_val_accuracy
                        if val_r2s[probe_name]:
                            wandb_logs[f"{probe_name}_val_r2"] = avg_val_r2
                        wandb.log(wandb_logs, step=epoch)
        
        return metrics
    
    def save_probes(self, output_dir):
        """
        Save the trained probes.
        
        Args:
            output_dir: Directory to save probes to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a dedicated weights directory
        weights_dir = os.path.join(output_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        # Save each probe
        for name, probe in self.probes.items():
            # Save state dict
            torch.save(probe.state_dict(), os.path.join(output_dir, f"{name}_probe.pt"))
            
            # Also save weights as NumPy arrays for easier analysis
            with torch.no_grad():
                weights = probe.linear.weight.data.clone().cpu().numpy()
                np.save(os.path.join(weights_dir, f"{name}_weights.npy"), weights)
                
                if hasattr(probe.linear, 'bias') and probe.linear.bias is not None:
                    bias = probe.linear.bias.data.clone().cpu().numpy()
                    np.save(os.path.join(weights_dir, f"{name}_bias.npy"), bias)
        
        # Save probe metadata
        metadata = {
            "hidden_dim": self.hidden_dim,
            "feature_dim": self.feature_dim,
            "layers_probed": self.layers_to_probe,
            "probe_names": list(self.probes.keys())
        }
        
        with open(os.path.join(output_dir, "probe_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved {len(self.probes)} probes to {output_dir}")
        logger.info(f"Saved probe weights to {weights_dir}")


def print_probe_weights(probes, output_dir):
    """
    Print and save information about the trained probe weights.
    
    Args:
        probes: Dictionary of trained probes
        output_dir: Directory to save weight information
    """
    logger.info("\nProbe weight analysis:")
    
    # Create a directory for weight information
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    weight_stats = {}
    
    for name, probe in probes.items():
        # Get the weights
        weights = probe.linear.weight.data.clone().cpu().numpy()
        
        # Calculate stats
        weight_stats[name] = {
            "shape": list(weights.shape),
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "l2_norm": float(np.linalg.norm(weights)),
        }
        
        # Print stats
        logger.info(f"\nProbe: {name}")
        logger.info(f"  Shape: {weights.shape}")
        logger.info(f"  Min: {weight_stats[name]['min']:.6f}")
        logger.info(f"  Max: {weight_stats[name]['max']:.6f}")
        logger.info(f"  Mean: {weight_stats[name]['mean']:.6f}")
        logger.info(f"  Std: {weight_stats[name]['std']:.6f}")
        logger.info(f"  L2 Norm: {weight_stats[name]['l2_norm']:.6f}")
        
        # Save weights to file (already done in save_probes, this is for redundancy)
        np.save(os.path.join(weights_dir, f"{name}_weights.npy"), weights)
    
    # Save stats as JSON
    with open(os.path.join(weights_dir, "weight_stats.json"), "w") as f:
        json.dump(weight_stats, f, indent=2)
    
    logger.info(f"\nSaved weight information to {weights_dir}")
    
    return weight_stats


class ProbeDatasetWrapper(Dataset):
    """
    Dataset wrapper for probe training.
    """
    def __init__(self, texts, labels, tokenizer, max_length=1024):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encodings = self.tokenizer(
            text, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to required format
        item = {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }
        
        return item


def load_custom_dataset(
    dataset_path, 
    text_field="text", 
    label_field="label",
    text_is_file=False,
    format="json",
    test_size=0.1
):
    """
    Load a custom dataset from a file or directory.
    
    Args:
        dataset_path: Path to dataset file or directory
        text_field: Field name for text data
        label_field: Field name for labels
        text_is_file: Whether text field contains file paths instead of text
        format: Format of the dataset file (json, csv, etc.)
        test_size: Fraction of data to use for testing
        
    Returns:
        train_texts, train_labels, val_texts, val_labels
    """
    if os.path.isfile(dataset_path):
        # Load from file
        if format.lower() == "json":
            import json
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        elif format.lower() == "csv":
            import pandas as pd
            data = pd.read_csv(dataset_path).to_dict('records')
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        texts = []
        labels = []
        
        for item in data:
            if text_field in item and label_field in item:
                text = item[text_field]
                
                # If text is a file path, load the file
                if text_is_file:
                    with open(text, 'r') as f:
                        text = f.read()
                
                # Process label - convert to float if it's a number
                label = item[label_field]
                if isinstance(label, (list, tuple)):
                    label = [float(l) for l in label]
                else:
                    label = float(label)
                
                texts.append(text)
                labels.append(label)
    elif os.path.isdir(dataset_path):
        # Load from directory
        raise ValueError("Directory-based datasets not yet implemented")
    else:
        raise ValueError(f"Could not find dataset at {dataset_path}")
    
    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    
    return train_texts, train_labels, val_texts, val_labels


def load_huggingface_dataset(
    dataset_name,
    text_field="text",
    label_field="label",
    split="train",
    test_size=0.1
):
    """
    Load a dataset from Hugging Face Datasets.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        text_field: Field name for text data
        label_field: Field name for labels
        split: Dataset split to use
        test_size: Fraction of data to use for testing
        
    Returns:
        train_texts, train_labels, val_texts, val_labels
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        # Extract texts and labels
        texts = []
        labels = []
        
        for item in dataset:
            if text_field in item and label_field in item:
                text = item[text_field]
                label = item[label_field]
                
                # Convert label to float array if it's a single value
                if not isinstance(label, (list, tuple)):
                    label = [float(label)]
                else:
                    label = [float(l) for l in label]
                    
                texts.append(text)
                labels.append(label)
        
        # Split into train and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        return train_texts, train_labels, val_texts, val_labels
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on model layers")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to use (path or Hugging Face dataset name)")
    parser.add_argument("--dataset_type", type=str, choices=["custom", "huggingface"], default="custom",
                        help="Type of dataset (custom file or Hugging Face)")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Field name for text data")
    parser.add_argument("--label_field", type=str, default="label",
                        help="Field name for labels")
    parser.add_argument("--format", type=str, default="json",
                        help="Format of custom dataset file")
    parser.add_argument("--output_dir", type=str, default="./linear_probes",
                        help="Directory to save probes to")
    parser.add_argument("--layers", type=str, default="0,5,10,15,20",
                        help="Comma-separated list of layer indices to probe")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for probe training")
    parser.add_argument("--feature_dim", type=int, default=1,
                        help="Dimension of feature space (number of outputs for probes)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="linear-probes",
                        help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--use_fp32", action="store_true",
                        help="Use FP32 precision instead of the default precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize W&B if requested
    if args.use_wandb:
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"probe-{os.path.basename(args.model_path)}",
                config=vars(args)
            )
            logger.info("Initialized W&B logging")
        except Exception as e:
            logger.warning(f"Error initializing W&B: {e}")
            args.use_wandb = False
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # If no pad token, use eos token as pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32 if args.use_fp32 else (torch.float16 if device.type == "cuda" else torch.float32),
        trust_remote_code=True
    ).to(device)
    
    # Put model in eval mode
    model.eval()
    
    # Parse layers to probe
    layer_indices = [int(idx) for idx in args.layers.split(",")]
    layers_to_probe = [f"layer_{idx}" for idx in layer_indices]
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    if args.dataset_type == "custom":
        train_texts, train_labels, val_texts, val_labels = load_custom_dataset(
            args.dataset,
            text_field=args.text_field,
            label_field=args.label_field,
            format=args.format
        )
    else:  # huggingface
        train_texts, train_labels, val_texts, val_labels = load_huggingface_dataset(
            args.dataset,
            text_field=args.text_field,
            label_field=args.label_field
        )
    
    # Make sure feature dimension matches label dimensions
    feature_dim = len(train_labels[0]) if isinstance(train_labels[0], (list, tuple)) else 1
    if args.feature_dim != feature_dim:
        logger.warning(f"Adjusted feature_dim from {args.feature_dim} to {feature_dim} to match labels")
    
    # Create datasets
    train_dataset = ProbeDatasetWrapper(
        train_texts, train_labels, tokenizer
    )
    
    val_dataset = ProbeDatasetWrapper(
        val_texts, val_labels, tokenizer
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Initialize trainer
    probe_trainer = ProbeTrainer(
        model=model,
        tokenizer=tokenizer,
        layers_to_probe=layers_to_probe,
        feature_dim=feature_dim,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Train probes
    logger.info("Starting probe training")
    metrics = probe_trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        use_wandb=args.use_wandb
    )
    
    # Save probes
    probe_trainer.save_probes(args.output_dir)
    
    # Print and save probe weight information
    weight_stats = print_probe_weights(probe_trainer.probes, args.output_dir)
    
    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create probe analysis plots (losses over time)
    try:
        import matplotlib.pyplot as plt
        
        # Create output directory for plots
        plots_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot loss curves 
        for layer_idx in layer_indices:
            probe_name = f"mlp_out_{layer_idx}"
            
            if probe_name in metrics:
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.plot(metrics[probe_name]['train_loss'], label='Train Loss')
                
                if val_dataloader and 'val_loss' in metrics[probe_name]:
                    plt.plot(metrics[probe_name]['val_loss'], label='Val Loss')
                
                plt.title(f"Loss Curves for {probe_name}")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                
                # Plot accuracy if available
                if 'train_accuracy' in metrics[probe_name] and len(metrics[probe_name]['train_accuracy']) > 0:
                    plt.subplot(2, 1, 2)
                    plt.plot(metrics[probe_name]['train_accuracy'], label='Train Accuracy')
                    
                    if 'val_accuracy' in metrics[probe_name] and len(metrics[probe_name]['val_accuracy']) > 0:
                        plt.plot(metrics[probe_name]['val_accuracy'], label='Val Accuracy')
                
                    plt.title(f"Accuracy for {probe_name}")
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.legend()
                # Plot R² if available and no accuracy
                elif 'train_r2' in metrics[probe_name] and len(metrics[probe_name]['train_r2']) > 0:
                    plt.subplot(2, 1, 2)
                    plt.plot(metrics[probe_name]['train_r2'], label='Train R²')
                    
                    if 'val_r2' in metrics[probe_name] and len(metrics[probe_name]['val_r2']) > 0:
                        plt.plot(metrics[probe_name]['val_r2'], label='Val R²')
                
                    plt.title(f"R² Score for {probe_name}")
                    plt.xlabel("Epoch")
                    plt.ylabel("R² Score")
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{probe_name}_metrics.png"))
                plt.close()
        
        # Plot layer comparison
        layer_indices = []
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_r2s = []
        val_r2s = []
        
        for layer_idx in sorted(layer_indices):
            probe_name = f"mlp_out_{layer_idx}"
            
            if probe_name in metrics:
                # Get final epoch metrics
                layer_indices.append(layer_idx)
                
                # Loss
                train_losses.append(metrics[probe_name]['train_loss'][-1] if metrics[probe_name]['train_loss'] else float('nan'))
                val_losses.append(metrics[probe_name]['val_loss'][-1] if metrics[probe_name]['val_loss'] else float('nan'))
                
                # Accuracy
                train_accuracies.append(metrics[probe_name]['train_accuracy'][-1] if 'train_accuracy' in metrics[probe_name] and metrics[probe_name]['train_accuracy'] else float('nan'))
                val_accuracies.append(metrics[probe_name]['val_accuracy'][-1] if 'val_accuracy' in metrics[probe_name] and metrics[probe_name]['val_accuracy'] else float('nan'))
                
                # R²
                train_r2s.append(metrics[probe_name]['train_r2'][-1] if 'train_r2' in metrics[probe_name] and metrics[probe_name]['train_r2'] else float('nan'))
                val_r2s.append(metrics[probe_name]['val_r2'][-1] if 'val_r2' in metrics[probe_name] and metrics[probe_name]['val_r2'] else float('nan'))
        
        if layer_indices:
            # Plot loss by layer
            plt.figure(figsize=(10, 8))
            plt.plot(layer_indices, train_losses, 'o-', label='Train Loss')
            plt.plot(layer_indices, val_losses, 's--', label='Val Loss')
            plt.xlabel("Layer Index")
            plt.ylabel("Final Loss")
            plt.title("Loss by Layer")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, "loss_by_layer.png"))
            plt.close()
            
            # Plot accuracy by layer if available
            if not all(np.isnan(x) for x in train_accuracies):
                plt.figure(figsize=(10, 8))
                plt.plot(layer_indices, train_accuracies, 'o-', label='Train Accuracy')
                plt.plot(layer_indices, val_accuracies, 's--', label='Val Accuracy')
                plt.xlabel("Layer Index")
                plt.ylabel("Final Accuracy")
                plt.title("Accuracy by Layer")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, "accuracy_by_layer.png"))
                plt.close()
            
            # Plot R² by layer if available
            if not all(np.isnan(x) for x in train_r2s):
                plt.figure(figsize=(10, 8))
                plt.plot(layer_indices, train_r2s, 'o-', label='Train R²')
                plt.plot(layer_indices, val_r2s, 's--', label='Val R²')
                plt.xlabel("Layer Index")
                plt.ylabel("Final R² Score")
                plt.title("R² Score by Layer")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, "r2_by_layer.png"))
                plt.close()
        
        logger.info(f"Created loss and performance plots in {plots_dir}")
    except Exception as e:
        logger.warning(f"Error creating plots: {e}")
    
    logger.info(f"Training complete. Probes saved to {args.output_dir}")
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()