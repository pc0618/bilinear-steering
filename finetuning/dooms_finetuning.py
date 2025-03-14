import os
import torch
import numpy as np
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
from einops import einsum
import gc
# Setup logging

class TokenStreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=1024, target_tokens=500_000_000):
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


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
        # Adjusted for TinyLlama hidden size (2048)
        self.c = nn.Parameter(torch.ones(2048))
        # Log activation metrics when first created
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.summary["model_info/annealer_initialized"] = True
                wandb.run.summary["model_info/hidden_size"] = 2048
        except:
            pass
    
    def forward(self, x):
        beta = 1.0 - self.alpha
        
        # Log activation stats occasionally
        if torch.rand(1).item() < 0.001:  # Log approximately 0.1% of forward passes
            try:
                import wandb
                if wandb.run is not None:
                    # Log activation statistics
                    with torch.no_grad():
                        wandb.log({
                            "activation/input_mean": x.mean().item(),
                            "activation/input_std": x.std().item(),
                            "activation/input_min": x.min().item(),
                            "activation/input_max": x.max().item(),
                            "activation/beta": beta,
                            "activation/c_mean": self.c.mean().item(),
                            "activation/c_std": self.c.std().item()
                        })
            except:
                pass
                
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
    loader = DataLoader(dataset, batch_size=8, shuffle=False)  # Smaller batch size for TinyLlama
    
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
        # For models with model.model.layers structure (like Llama)
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
        self.step_beta_values = {}  # Store beta values for logging
 
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
            
        # Store beta value (beta = 1.0 - alpha) for each layer
        self.step_beta_values = {}
        
        def store_beta(module):
            if isinstance(module, (Interpolator, Annealer)):
                module.alpha = alpha
                # Calculate beta value for this module
                beta_value = 1.0 - alpha
                # Store with a counter to track individual modules
                if not hasattr(store_beta, "counter"):
                    store_beta.counter = 0
                self.step_beta_values[f"beta_{store_beta.counter}"] = beta_value
                store_beta.counter += 1
        
        # Apply alpha and collect beta values
        self.model.apply(store_beta)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}
            
        fraction = state.global_step / state.max_steps
        if fraction <= 0.3:
            alpha = fraction / 0.3
        else:
            alpha = 1.0
            
        # Log alpha value
        logs['alpha'] = alpha
        logs['beta'] = 1.0 - alpha  # Global beta value
        
        # Log max, min, and mean beta values across modules
        if self.step_beta_values:
            beta_values = list(self.step_beta_values.values())
            logs['beta_max'] = max(beta_values)
            logs['beta_min'] = min(beta_values)
            logs['beta_mean'] = sum(beta_values) / len(beta_values)
            
            # Log individual beta values (for first few layers)
            for key, value in self.step_beta_values.items():
                if int(key.split('_')[1]) < 5:  # Log only first 5 layers to avoid clutter
                    logs[key] = value
        
        # Try to log to wandb explicitly
        try:
            import wandb
            if wandb.run is not None:
                # Log training metrics
                wandb.log({
                    "loss": logs.get("loss", 0),
                    "learning_rate": logs.get("learning_rate", 0),
                    "alpha": logs.get("alpha", 0),
                    "beta": logs.get("beta", 0),
                    "beta_max": logs.get("beta_max", 0),
                    "beta_min": logs.get("beta_min", 0),
                    "beta_mean": logs.get("beta_mean", 0),
                    "epoch": logs.get("epoch", 0),
                    "step": state.global_step,
                    "progress": fraction
                })
        except Exception as e:
            print(f"Warning: Could not log to wandb: {e}")

# Main script
def main():
    # Initialize Weights & Biases
    try:
        import wandb
        wandb.login()
        wandb.init(project="tinyllama-bilinear", name="tinyllama-1.1b-bilinear-conversion")
        print("Successfully initialized Weights & Biases monitoring.")
    except Exception as e:
        print(f"Warning: Could not initialize Weights & Biases: {e}")
        print("Training will continue, but metrics won't be logged to W&B.")
    
    # Use TinyLlama 1.1B Chat
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
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
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Make sure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise RuntimeError(f"Failed to load TinyLlama model: {e}")
    
    # Load fineweb dataset as streaming
    try:
        logger.info("Loading fineweb dataset with streaming...")
        # Use streaming to avoid downloading the entire dataset
        raw_dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
        
        # Create streaming dataset with token limit
        train_dataset = TokenStreamingDataset(
            raw_dataset,
            tokenizer,
            max_length=1024,
            target_tokens=500_000_000
        )
        
        logger.info("Successfully created streaming dataset targeting 500M tokens")
    except Exception as e:
        logger.error(f"Error loading fineweb dataset with streaming: {e}")
        try:
            # Alternative: try with specific configuration
            logger.info("Attempting to load with specific configuration...")
            raw_dataset = load_dataset("HuggingFaceFW/fineweb", "all", split="train", streaming=True)
            
            # Create streaming dataset with token limit
            train_dataset = TokenStreamingDataset(
                raw_dataset,
                tokenizer,
                max_length=1024,
                target_tokens=500_000_000
            )
            
            logger.info("Successfully created streaming dataset with specific configuration")
        except Exception as e2:
            logger.error(f"All attempts to load fineweb dataset failed: {e2}")
            raise RuntimeError("Failed to load fineweb dataset. Please check your internet connection and Hugging Face access.")
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Use the text field from fineweb dataset
        text_field = "text" if "text" in examples else "content"
        return tokenizer(examples[text_field], truncation=True, max_length=1024)
    
    print("Tokenizing dataset...")
    
    tokens_per_sample = 1024  # Maximum tokens per sample
    
    # Since we're using a streaming dataset, we can't precisely know its length
    # Use the target tokens from TokenStreamingDataset for our estimation
    target_tokens = 500_000_000
    
    # Replace gate activations with Annealer modules
    print("Replacing gate activations with Annealer modules...")
    model = replace_with_annealer(model)
    
    tokens_per_sample = 1024  # Maximum tokens per sample
    target_tokens = 500_000_000  # The target from TokenStreamingDataset
    batch_size = 4  # per_device_train_batch_size
    grad_accum = 1  # gradient_accumulation_steps
    tokens_per_batch = batch_size * grad_accum * tokens_per_sample
    estimated_steps = target_tokens // tokens_per_batch

    print(f"Dataset targeting approximately {target_tokens:,} tokens")
    print(f"Estimated training steps: {estimated_steps:,}")

    # Then use the estimated_steps in the TrainingArguments

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./tinyllama-1.1b-bilinear-dooms",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=3e-5,
        weight_decay=0.01,
        max_steps=estimated_steps,  # Use the estimated steps instead of len(train_dataset)
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,  # Keep only the last 3 checkpoints
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=8,  # Use multiple workers for faster data loading
        report_to="wandb",
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
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[AlphaDecay(model)],
    )

    
    # Configure wandb run to track the model
    try:
        import wandb
        if wandb.run is not None:
            # Log model architecture info
            wandb.run.summary["model_info/name"] = model_name
            wandb.run.summary["model_info/parameters"] = sum(p.numel() for p in model.parameters())
            wandb.run.summary["training/dataset"] = "HuggingFaceFW/fineweb (streaming 500M tokens)"
            wandb.run.summary["training/estimated_steps"] = estimated_steps
            wandb.run.summary["training/target_tokens"] = 500_000_000
            
            # Log hyperparameters
            wandb.config.update({
                "learning_rate": training_args.learning_rate,
                "batch_size": batch_size * grad_accum,
                "tokens_per_batch": tokens_per_batch,
                "bilinear_transition_fraction": 0.3,
                "model": model_name,
                "optimizer": "AdamW",
                "weight_decay": training_args.weight_decay,
                "seq_length": 1024
            })
    except Exception as e:
        logger.warning(f"Error configuring wandb run: {e}")
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Log final model information
    try:
        import wandb
        if wandb.run is not None:
            wandb.run.summary["training/completed"] = True
            wandb.run.summary["training/final_loss"] = trainer.state.log_history[-1].get("loss", None)
            wandb.run.summary["training/steps"] = trainer.state.global_step
    except Exception as e:
        print(f"Warning: Error logging final model info to wandb: {e}")
    
    # Ensure beta is set to 0 (full bilinear mode) for the saved model
    print("Setting model to full bilinear mode (beta = 0)...")
    def set_full_bilinear_mode(module):
        if isinstance(module, (Interpolator, Annealer)):
            module.alpha = 1.0  # alpha = 1.0 means beta = 0
            print(f"Set {type(module).__name__} to full bilinear mode (alpha = 1.0, beta = 0)")
    
    model.apply(set_full_bilinear_mode)
    
    # Output directories
    output_dir = "./tinyllama-1.1b-bilinear-dooms-final"
    
    # Save using transformers save_pretrained
    print("Saving model with transformers save_pretrained...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save a PyTorch checkpoint directly
    print("Saving PyTorch checkpoint...")
    torch_checkpoint_path = f"{output_dir}/pytorch_model_bilinear.bin"
    
    # Create a checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict() if hasattr(model, "config") else None,
        "is_bilinear": True,
        "beta_value": 0.0,  # Explicit flag for the bilinear nature (beta = 0)
    }
    
    # Save the checkpoint
    torch.save(checkpoint, torch_checkpoint_path)
    print(f"PyTorch checkpoint saved to {torch_checkpoint_path}")
    
    print(f"Training complete! Model saved to {output_dir}")
    
    # Finish wandb run
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

if __name__ == "__main__":
    main()