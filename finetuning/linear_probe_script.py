import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import argparse
import logging
import json
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb
from hook_solution import InterpolatedSiLU  # Import your custom activation class
from bilinear_inference import register_custom_modules, load_bilinear_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("linear_probe.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LinearProbe(nn.Module):
    """Linear probe model that maps from hidden states to target features"""
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is not None:
            # Two-layer MLP for more complex mappings
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Simple linear layer
            self.model = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.model(x)

class ResidualStreamHook:
    """Hook to capture residual stream activations at specified layers"""
    def __init__(self, model, layer_names=None):
        self.model = model
        self.activations = {}
        self.handles = []
        
        # If layer_names not provided, extract all MLP layers
        if layer_names is None:
            layer_names = self._find_mlp_layers()
        
        self.layer_names = layer_names
        logger.info(f"Registering hooks for {len(layer_names)} layers")
        
        # Register hooks for each layer
        for name in layer_names:
            self._register_hook(name)
    
    def _find_mlp_layers(self):
        """Find all MLP layers in the model"""
        mlp_layers = []
        
        # First try to find layers by direct name match
        for name, _ in self.model.named_modules():
            if ".mlp" in name and name.endswith(".mlp"):
                mlp_layers.append(name)
        
        # If no layers found, try a more general approach
        if not mlp_layers:
            logger.info("No mlp layers found by exact name, trying more general patterns")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList) and "layers" in name:
                    for i, layer in enumerate(module):
                        if hasattr(layer, "mlp"):
                            mlp_layers.append(f"{name}.{i}.mlp")
        
        logger.info(f"Found {len(mlp_layers)} MLP layers")
        return mlp_layers
    
    def _register_hook(self, layer_name):
        """Register a forward hook for a specific layer"""
        # Find the module
        module = self.model
        for part in layer_name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        # Define hook function to capture output
        def hook_fn(module, input, output):
            # Store the output (which contributes to the residual stream)
            self.activations[layer_name] = output
        
        # Register the hook
        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        
    def clear(self):
        """Clear stored activations"""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []

class FeatureDataset(Dataset):
    """Generic dataset for linear probe training with activations and targets"""
    def __init__(self, activations, targets):
        self.activations = activations
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]

def extract_activations(model, tokenizer, texts, layer_names=None, batch_size=8, max_length=512):
    """
    Extract residual stream activations for a set of texts
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of text samples
        layer_names: List of layer names to extract activations from (or None for all MLP layers)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
    
    Returns:
        all_activations: Dictionary mapping layer names to activation tensors
    """
    logger.info(f"Extracting activations for {len(texts)} samples")
    
    # Save original dtype
    original_dtype = next(model.parameters()).dtype
    
    # Register hooks
    try:
        hook = ResidualStreamHook(model, layer_names)
    except Exception as e:
        logger.error(f"Error registering hooks: {str(e)}")
        # Fallback to manually finding layers
        if layer_names is None:
            layer_names = []
            for name, module in model.named_modules():
                if ".mlp" in name and name.endswith(".mlp"):
                    layer_names.append(name)
            logger.info(f"Manually found {len(layer_names)} MLP layers")
        
        hook = ResidualStreamHook(model, layer_names)
    
    # Process in batches
    all_activations = {layer: [] for layer in hook.layer_names}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(model.device)
            
            # Forward pass - we only need this to trigger the hooks
            model(**inputs)
            
            # Store activations for this batch
            for layer_name, activation in hook.activations.items():
                # Pool over all tokens in the sequence (mean pooling)
                # This captures information from the entire sequence rather than just the final token
                pooled_activations = activation.mean(dim=1)  # Average across sequence dimension
                
                # Store as float32 to avoid dtype issues during training
                all_activations[layer_name].append(pooled_activations.cpu().float())
            
            # Clear activations for next batch
            hook.clear()
    
    # Remove hooks when done
    hook.remove_hooks()
    
    # Concatenate results
    for layer_name in all_activations:
        all_activations[layer_name] = torch.cat(all_activations[layer_name], dim=0)
    
    logger.info(f"Extracted activations from {len(all_activations)} layers")
    return all_activations

# In paste-2.txt (training script), modify these functions:

def train_linear_probe(activations, targets, val_split=0.1, batch_size=32, learning_rate=1e-3, 
                       hidden_dim=None, num_epochs=5, weight_decay=0, task_type='classification'):
    """
    Train a linear probe on layer activations
    
    Args:
        activations: Tensor of activations (batch_size, hidden_dim)
        targets: Tensor of target values/labels
        val_split: Fraction of data to use for validation
        batch_size: Training batch size
        learning_rate: Learning rate
        hidden_dim: If provided, use a 2-layer MLP instead of a linear layer
        num_epochs: Number of training epochs
        weight_decay: L2 regularization strength
        task_type: 'classification' or 'regression'
    
    Returns:
        probe: Trained LinearProbe model
        metrics: Dictionary of evaluation metrics
    """
    # Convert activations to float32 to ensure dtype compatibility
    activations = activations.float()
    
    # Create dataset
    dataset = FeatureDataset(activations, targets)
    
    # Split into train/val
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Determine output dimension - MODIFIED for binary classification
    if task_type == 'classification':
        num_classes = len(torch.unique(targets))
        # Use 1 output dimension for binary classification
        output_dim = 1 if num_classes == 2 else num_classes
    else:  # regression
        output_dim = 1
    
    # Create model
    input_dim = activations.shape[1]
    probe = LinearProbe(input_dim, output_dim, hidden_dim)
    
    # Loss function - MODIFIED to use BCEWithLogitsLoss for binary classification
    if task_type == 'classification':
        if output_dim == 1:  # Binary classification
            criterion = nn.BCEWithLogitsLoss()
        else:  # Multi-class classification
            criterion = nn.CrossEntropyLoss()
    else:  # regression
        criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    best_loss = float('inf')
    best_probe = None
    
    for epoch in range(num_epochs):
        # Training
        probe.train()
        train_loss = 0
        
        for batch_activations, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = probe(batch_activations)
            
            # MODIFIED: Handle binary classification with 1D output
            if task_type == 'classification' and output_dim == 1:
                # Convert targets to float for BCEWithLogitsLoss
                outputs = outputs.squeeze()
                batch_targets = batch_targets.float()
            elif task_type == 'regression' and outputs.shape[1] == 1:
                outputs = outputs.squeeze()
            
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        probe.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_activations, batch_targets in val_loader:
                outputs = probe(batch_activations)
                
                # MODIFIED: Handle binary classification with 1D output
                if task_type == 'classification' and output_dim == 1:
                    outputs = outputs.squeeze()
                    batch_targets_loss = batch_targets.float()
                    loss = criterion(outputs, batch_targets_loss)
                    # Apply threshold for predictions
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                elif task_type == 'classification':
                    loss = criterion(outputs, batch_targets)
                    preds = torch.argmax(outputs, dim=1)
                else:  # regression
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_targets)
                    preds = outputs
                
                val_loss += loss.item()
                all_preds.append(preds)
                all_targets.append(batch_targets)
        
        val_loss /= len(val_loader)
        
        # Update best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_probe = probe.state_dict()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Load best model
    probe.load_state_dict(best_probe)
    
    # Calculate final metrics
    probe.eval()
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = {"val_loss": best_loss}
    
    if task_type == 'classification':
        metrics["accuracy"] = accuracy_score(all_targets.numpy(), all_preds.numpy())
        metrics["f1"] = f1_score(all_targets.numpy(), all_preds.numpy(), average='weighted')
        metrics["precision"] = precision_score(all_targets.numpy(), all_preds.numpy(), average='weighted')
        metrics["recall"] = recall_score(all_targets.numpy(), all_preds.numpy(), average='weighted')
    else:  # regression
        metrics["mse"] = ((all_preds - all_targets) ** 2).mean().item()
    
    return probe, metrics

def analyze_probe_weights(probe, feature_names=None, top_k=10):
    """
    Analyze the weights of a linear probe to identify important features
    
    Args:
        probe: Trained LinearProbe model
        feature_names: Optional list of feature names
        top_k: Number of top features to return
    
    Returns:
        top_features: List of top feature indices or names
        weights: Corresponding weight values
    """
    # Extract weights from the probe (first layer if it's an MLP)
    if isinstance(probe.model, nn.Sequential):
        weights = probe.model[0].weight.data
    else:
        weights = probe.model.weight.data
    
    # MODIFIED: For binary classification, we now have a 1D weight vector
    # So we can just take the absolute value directly
    if weights.dim() == 2 and weights.shape[0] == 1:
        # Binary classification with 1D output - reshape to 1D
        importance = weights.abs().squeeze()
    elif weights.dim() > 1 and weights.shape[0] > 1:
        # Multi-class case - compute L2 norm across classes
        importance = torch.norm(weights, dim=0)
    else:
        # Already 1D vector (regression or binary)
        importance = weights.abs()
    
    # Get top-k indices
    top_indices = torch.argsort(importance, descending=True)[:top_k].cpu().numpy()
    
    # Get corresponding weights
    top_weights = importance[top_indices].cpu().numpy()
    
    # Map to feature names if provided
    if feature_names is not None:
        top_features = [feature_names[i] for i in top_indices]
    else:
        top_features = top_indices
    
    return top_features, top_weights

def train_probes_all_layers(model, tokenizer, texts, targets, layer_names=None, output_dir="./probes",
                           task_type='classification', **train_kwargs):
    """
    Train linear probes for each layer's residual stream
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of text samples
        targets: Tensor of target values/labels
        layer_names: List of layer names to extract activations from (or None for all MLP layers)
        output_dir: Directory to save probes and results
        task_type: 'classification' or 'regression'
        train_kwargs: Additional arguments for train_linear_probe
    
    Returns:
        results: Dictionary of results for each layer
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract activations for all layers
    layer_activations = extract_activations(model, tokenizer, texts, layer_names)
    
    # Convert targets to tensor if it's not already
    if not isinstance(targets, torch.Tensor):
        if task_type == 'classification':
            targets = torch.tensor(targets, dtype=torch.long)
        else:  # regression
            targets = torch.tensor(targets, dtype=torch.float)
    
    # Train a probe for each layer
    results = {}
    for layer_name, activations in layer_activations.items():
        logger.info(f"Training probe for layer: {layer_name}")
        
        # Train probe
        probe, metrics = train_linear_probe(
            activations, targets, task_type=task_type, **train_kwargs
        )
        
        # Save probe
        layer_dir = os.path.join(output_dir, layer_name.replace(".", "_"))
        os.makedirs(layer_dir, exist_ok=True)
        torch.save(probe.state_dict(), os.path.join(layer_dir, "probe.pt"))
        
        # Save metrics
        with open(os.path.join(layer_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Store results
        results[layer_name] = {
            "metrics": metrics,
            "probe_path": os.path.join(layer_dir, "probe.pt")
        }
        
        # Optional: analyze probe weights
        if isinstance(probe.model, nn.Linear) or isinstance(probe.model, nn.Sequential):
            top_features, weights = analyze_probe_weights(probe)
            results[layer_name]["top_features"] = top_features.tolist() if isinstance(top_features, np.ndarray) else top_features
            results[layer_name]["top_weights"] = weights.tolist()
    
    # Save overall results
    with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
        # Convert any numpy arrays to lists for JSON serialization
        serializable_results = json.loads(json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
        json.dump(serializable_results, f, indent=2)
    
    return results

class FeatureDataProcessor:
    """Process different types of datasets for feature probing"""
    
    @staticmethod
    def load_sentiment_dataset(dataset_path, split="train", sample_size=None):
        """Load a sentiment analysis dataset (e.g., SST-2)"""
        from datasets import load_dataset
        
        try:
            # Try loading from HuggingFace datasets
            dataset = load_dataset("sst2", split=split)
        except Exception:
            # Try loading from local path
            dataset = load_dataset(dataset_path, split=split)
        
        # Sample if needed
        if sample_size is not None and sample_size < len(dataset):
            dataset = dataset.select(range(sample_size))
        
        texts = dataset["sentence"]
        labels = dataset["label"]
        
        return texts, labels
    
    @staticmethod
    def load_toxicity_dataset(dataset_path, split="train", sample_size=None):
        """Load a toxicity detection dataset"""
        from datasets import load_dataset
        
        try:
            # Try loading from HuggingFace datasets
            dataset = load_dataset("civil_comments", split=split)
            texts = dataset["text"]
            labels = (dataset["toxicity"] >= 0.5).astype(int)  # Binarize toxicity scores
        except Exception:
            # Try loading from local path or another dataset
            dataset = load_dataset(dataset_path, split=split)
            texts = dataset["text"]
            if "toxicity" in dataset.features:
                labels = (dataset["toxicity"] >= 0.5).astype(int)
            else:
                labels = dataset["label"]
        
        # Sample if needed
        if sample_size is not None and sample_size < len(dataset):
            indices = np.random.choice(len(dataset), size=sample_size, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        return texts, labels
    
    @staticmethod
    def load_custom_dataset(file_path, text_col="sentences", label_col="sentiment", sample_size=None):
        """Load a custom dataset from CSV or JSON with modified handling for sentiment data"""
        import pandas as pd
        
        # Determine file type and load
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Sample if needed
        if sample_size is not None and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
        
        # Handle the specific case of sentiment_prompts.csv which has 'sentences' and 'sentiment' columns
        if 'sentences' in df.columns and 'sentiment' in df.columns:
            texts = df['sentences'].tolist()
            labels = df['sentiment'].tolist()
        else:
            # Fall back to the provided column names
            texts = df[text_col].tolist()
            labels = df[label_col].tolist()
        
        return texts, labels
    
    @staticmethod
    def load_few_shot_samples(samples_file):
        """Load few-shot examples from a JSON file"""
        with open(samples_file, 'r') as f:
            samples = json.load(f)
        
        texts = [sample['text'] for sample in samples]
        labels = [sample['label'] for sample in samples]
        
        return texts, labels

def main():
    """Main function to run the linear probe training"""
    parser = argparse.ArgumentParser(description="Train linear probes on TinyLlama model residual stream")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--dataset_type", type=str, default="sentiment", 
                        choices=["sentiment", "toxicity", "custom", "few_shot"],
                        help="Type of dataset to use")
    parser.add_argument("--dataset_path", type=str, default=None, 
                        help="Path to dataset or name of HuggingFace dataset")
    parser.add_argument("--text_col", type=str, default="sentences", 
                        help="Column name for text in custom dataset")
    parser.add_argument("--label_col", type=str, default="sentiment", 
                        help="Column name for labels in custom dataset")
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Number of samples to use from dataset")
    parser.add_argument("--output_dir", type=str, default="./linear_probes",
                        help="Directory to save probes and results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for activation extraction and training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for probe training")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Hidden dimension for MLP probe (None for linear probe)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--task_type", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Task type: classification or regression")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log results to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="tinyllama-linear-probes",
                        help="Weights & Biases project name")
    parser.add_argument("--layer_pattern", type=str, default=None,
                        help="Regex pattern to filter layer names (optional)")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Register custom modules first
    register_custom_modules()
    
    # Load model and tokenizer using the robust bilinear model loader
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_bilinear_model(args.model_path)
    
    # Prepare dataset
    logger.info(f"Loading {args.dataset_type} dataset")
    
    if args.dataset_type == "sentiment":
        texts, labels = FeatureDataProcessor.load_sentiment_dataset(
            args.dataset_path, sample_size=args.sample_size
        )
    elif args.dataset_type == "toxicity":
        texts, labels = FeatureDataProcessor.load_toxicity_dataset(
            args.dataset_path, sample_size=args.sample_size
        )
    elif args.dataset_type == "custom":
        texts, labels = FeatureDataProcessor.load_custom_dataset(
            args.dataset_path, args.text_col, args.label_col, sample_size=args.sample_size
        )
    elif args.dataset_type == "few_shot":
        if not args.dataset_path:
            raise ValueError("Dataset path must be provided for few-shot samples")
        texts, labels = FeatureDataProcessor.load_few_shot_samples(args.dataset_path)
    
    # Filter layers if pattern provided
    layer_names = None
    if args.layer_pattern:
        import re
        pattern = re.compile(args.layer_pattern)
        layer_names = []
        
        for name, _ in model.named_modules():
            if pattern.search(name):
                layer_names.append(name)
        
        logger.info(f"Selected {len(layer_names)} layers based on pattern '{args.layer_pattern}'")
    
    # Train probes for all layers
    results = train_probes_all_layers(
        model,
        tokenizer,
        texts,
        labels,
        layer_names=layer_names,
        output_dir=args.output_dir,
        task_type=args.task_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        val_split=args.val_split
    )
    
    # Log to wandb if enabled
    if args.use_wandb:
        # Log summary metrics for each layer
        for layer_name, layer_results in results.items():
            for metric_name, value in layer_results["metrics"].items():
                wandb.log({f"{layer_name}/{metric_name}": value})
        
        # Create visualization of metrics across layers
        layer_names = list(results.keys())
        if args.task_type == "classification":
            metric_name = "accuracy"
        else:
            metric_name = "mse"
        
        metrics = [results[layer]["metrics"][metric_name] for layer in layer_names]
        
        # Create a summary dataframe and log it
        import pandas as pd
        summary_df = pd.DataFrame({
            "layer": layer_names,
            metric_name: metrics
        })
        
        wandb.log({"layer_performance": wandb.Table(dataframe=summary_df)})
        wandb.finish()
    
    logger.info(f"Probe training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise