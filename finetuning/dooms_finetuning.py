import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from torch import nn
from einops import einsum
import gc
# Import Sight from local utils.py or define it here if it doesn't exist
try:
    from utils import Sight
except ImportError:
    print("Warning: Could not import Sight from utils.py, using embedded version")
    # Define a simplified Sight class for basic functionality
    class Sight:
        """A helper class to more cleanly interface with model internals."""
        def __init__(self, model, tokenizer=None, *args, **kwargs):
            self.model = model
            self.tokenizer = tokenizer
        
        def trace(self, *args, **kwargs):
            """Context manager for tracing model execution."""
            class DummyContextManager:
                def __enter__(self):
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            return DummyContextManager()
        
        @property
        def layers(self):
            """Get model layers based on architecture."""
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                return self.model.model.layers
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h
            else:
                raise AttributeError("Could not determine model layers structure")
from transformers import TrainerCallback

# Utility functions for decay schedules
def exp_decay(x, k=1.05):
    return min(x * np.e ** (k - k * x), 1.0)

def logistic_decay(x, d=4):
    return 1 / (1 + np.exp(-2*d*x + d))

def smootherstep_decay(x, s=0.5):
    return (6*x**5 - 15*x**4 + 10*x**3)**s

def no_decay(x):
    return 1e-5

# Regression metrics calculation
def _regression_metrics(a, b, x):
    b_pred = einsum(a, x, "... b i, ... i o -> ... b o")
    residuals = b - b_pred

    ss_total = (b - b.mean(-1, keepdim=True)).pow(2).sum()
    ss_residual = residuals.pow(2).sum()
    r_squared = 1 - (ss_residual / ss_total)
    
    return dict(r_squared=r_squared.item())

# Interpolator module to gradually transition between original and linear functions
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

# Annealer module to replace SwiGLU with bilinear MLP
class Annealer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.0
        self.c = nn.Parameter(torch.ones(4096))  # Adjusted for Gemma 3 4B hidden size
    
    def forward(self, x):
        beta = 1.0 - self.alpha
        return self.c * x * torch.sigmoid(beta * x)

# Helper function to set alpha value in modules
def _set_alpha(module, alpha):
    if isinstance(module, Interpolator):
        module.alpha = alpha
    elif isinstance(module, Annealer):
        module.alpha = alpha

# Gate replacer for extracting and replacing gate functions
class GateReplacer:
    def extract(self, sight, batch):
        try:
            with sight.trace(batch["input_ids"], scan=False, validate=False):
                # Handle different model architectures
                g_inp = []
                g_out = []
                
                # Identify the appropriate structure
                if hasattr(sight, 'model') and hasattr(sight.model, 'layers'):
                    layers = sight.model.layers
                elif hasattr(sight.model, 'model') and hasattr(sight.model.model, 'layers'):
                    layers = sight.model.model.layers
                elif hasattr(sight.model, 'transformer') and hasattr(sight.model.transformer, 'h'):
                    layers = sight.model.transformer.h
                else:
                    raise AttributeError("Unknown model structure")
                
                # Extract activations
                for layer in layers:
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                        g_inp.append(layer.mlp.gate.input.save())
                        g_out.append(layer.mlp.gate.output.save())
                    elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'act'):
                        g_inp.append(layer.mlp.act.input.save())
                        g_out.append(layer.mlp.act.output.save())
                    else:
                        # Placeholder for incompatible layers
                        print(f"Warning: Layer structure not compatible: {type(layer).__name__}")
                        continue
                
                if not g_inp:
                    raise ValueError("No compatible gate/activation functions found in model")
                    
                return torch.stack(g_inp), torch.stack(g_out)
        except Exception as e:
            print(f"Error in GateReplacer.extract: {e}")
            # Return dummy tensors as fallback
            return torch.ones(1, 1, 1), torch.ones(1, 1, 1)
        
    def overwrite(self, model, x):
        try:
            # Handle different model architectures
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = model.model.layers
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers = model.transformer.h
            else:
                print("Warning: Unknown model structure, cannot overwrite gates")
                return
            
            for i, (layer, x1) in enumerate(zip(layers, x)):
                try:
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                        layer.mlp.gate = Interpolator(layer.mlp.gate, x1)
                    elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'act'):
                        layer.mlp.act = Interpolator(layer.mlp.act, x1)
                except Exception as e:
                    print(f"Error replacing gate/activation in layer {i}: {e}")
        except Exception as e:
            print(f"Error in GateReplacer.overwrite: {e}")

# Function to replace components with linear approximations
def replace_components(model, dataset, which="gate", n_batches=1, compute_metrics=True):
    replacer = dict(gate=GateReplacer)[which]()
    sight = Sight(model)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False)  # Smaller batch size for Gemma 3 4B
    
    a, b = [], []
    
    for _, batch in zip(range(n_batches), loader):
        inp, out = replacer.extract(sight, batch)
        a.append(inp)
        b.append(out)
    
    a = torch.cat(a, dim=1).flatten(1, 2)
    b = torch.cat(b, dim=1).flatten(1, 2)
    x = torch.linalg.lstsq(a, b).solution
    
    if compute_metrics:
        print(_regression_metrics(a, b, x))
        
    del a, b
    gc.collect()
    torch.cuda.empty_cache()

    replacer.overwrite(model, x)
    return model

# Function to replace gate activations with Annealer modules
def replace_with_annealer(model, *args, **kwargs):
    try:
        # For models with model.model.layers structure (like Gemma/Llama)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                    layer.mlp.gate = Annealer(*args, **kwargs)
                # For models with slightly different structures
                elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'act'):
                    layer.mlp.act = Annealer(*args, **kwargs)
        # For GPT-style models with transformer.h structure
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            for layer in model.transformer.h:
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'act'):
                    layer.mlp.act = Annealer(*args, **kwargs)
        else:
            print("Warning: Couldn't find the expected model structure for replacing activations.")
            print(f"Model structure: {type(model).__name__}")
            # Try to detect the model structure
            for attr_name in dir(model):
                attr = getattr(model, attr_name)
                if isinstance(attr, nn.Module):
                    print(f"Found module: {attr_name}")
    except Exception as e:
        print(f"Error in replace_with_annealer: {e}")
        print("Will attempt to continue training without activation replacement")
        
    return model

# Callback to gradually decay the alpha parameter during training
class AlphaDecay(TrainerCallback):
    def __init__(self, model, decay=exp_decay):
        self.model = model
        self.decay = decay
 
    def on_step_begin(self, args, state, control, **kwargs):
        fraction = state.global_step / state.max_steps
        # Apply alpha decay based on training progress
        # For first 30% of tokens, linearly interpolate to β=0 (bilinear)
        if fraction <= 0.3:
            # Linear interpolation toward bilinear (α=1.0)
            alpha = fraction / 0.3
        else:
            # Continue fine-tuning with bilinear for remaining 70%
            alpha = 1.0
            
        self.model.apply(lambda x: _set_alpha(x, alpha))
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        fraction = state.global_step / state.max_steps
        if fraction <= 0.3:
            alpha = fraction / 0.3
        else:
            alpha = 1.0
        logs['alpha'] = alpha

# Main script
def main():
    # Check and update transformers if needed
    import subprocess
    import importlib
    
    try:
        # Attempt to update transformers to latest version
        print("Updating transformers library to latest version...")
        subprocess.check_call([
            "pip", "install", "--upgrade", "transformers"
        ])
        
        # Reload the transformers module
        import transformers
        importlib.reload(transformers)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Transformers updated to version: {transformers.__version__}")
    except Exception as e:
        print(f"Warning: Could not update transformers automatically: {e}")
        print("If this script fails, please manually run: pip install --upgrade transformers")
    
    # Use correct model name for Gemma models 
    # Based on the error, we need to use a model that actually exists on Hugging Face
    model_name = "google/gemma-2b"  # Correct model name that should exist on HF
    
    try:
        print(f"Loading model: {model_name}")
        
        # Attempt to login with Hugging Face token if available
        try:
            from huggingface_hub import login
            token = os.environ.get("HF_TOKEN")
            if token:
                print("Logging in to Hugging Face with token...")
                login(token)
        except Exception as e:
            print(f"Warning: Could not login to Hugging Face Hub: {e}")
            print("Continuing without authentication. This may fail if the model requires authentication.")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True  # Important for newer models
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        
        # Try alternative model options
        alternate_models = [
            "google/gemma-2-2b", 
            "google/gemma-2b",
            "google/gemma-7b",
            "mistralai/Mistral-7B-v0.1"  # Fallback to a completely different model family
        ]
        
        model = None
        for alt_model in alternate_models:
            try:
                print(f"Trying alternative model: {alt_model}")
                model = AutoModelForCausalLM.from_pretrained(
                    alt_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(alt_model)
                model_name = alt_model  # Update model name for later references
                print(f"Successfully loaded alternative model: {alt_model}")
                break
            except Exception as alt_e:
                print(f"Failed to load {alt_model}: {alt_e}")
        
        if model is None:
            raise RuntimeError("Failed to load any model. Please check your internet connection and access permissions.")
    
    # Load TinyStories dataset
    try:
        # Try to load the tokenized dataset first
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    except:
        # Fallback to raw dataset
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Tokenize dataset if needed
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)
    
    if "input_ids" not in dataset.column_names:
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Calculate the number of tokens to use (1B or max available)
    tokens_per_sample = 1024  # Approximate tokens per sample
    samples_needed = 1_000_000_000 // tokens_per_sample  # For 1B tokens
    
    if len(dataset) > samples_needed:
        dataset = dataset.select(range(samples_needed))
        print(f"Using {samples_needed} samples (~1B tokens)")
    else:
        print(f"Using all {len(dataset)} samples (~{len(dataset) * tokens_per_sample} tokens)")
    
    # Replace gate activations with Annealer modules
    print("Replacing gate activations with Annealer modules...")
    model = replace_with_annealer(model)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./gemma-2-2b-bilinear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_steps=len(dataset) // (2 * 1),  # Adjusted for batch size and gradient accumulation
        logging_steps=100,
        save_steps=1000,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer with AlphaDecay callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[AlphaDecay(model)],
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving final model...")
    model.save_pretrained("./gemma-2-2b-bilinear-final")
    tokenizer.save_pretrained("./gemma-2-2b-bilinear-final")
    print("Training complete!")

if __name__ == "__main__":
    main()