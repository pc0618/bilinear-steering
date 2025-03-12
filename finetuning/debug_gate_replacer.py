import torch
from torch import nn
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llama_debug")

class DebugGateReplacer:
    def __init__(self):
        self.logger = logger
    
    def inspect_model_architecture(self, model):
        """Inspect and log the model architecture to understand its structure"""
        self.logger.info("Inspecting model architecture...")
        
        # Check the main model attributes
        model_attrs = dir(model)
        self.logger.info(f"Main model attributes: {[attr for attr in model_attrs if not attr.startswith('_')]}")
        
        # Check if it's a Llama model
        if hasattr(model, 'model'):
            self.logger.info("Found 'model' attribute - likely a Llama model")
            if hasattr(model.model, 'layers'):
                self.logger.info(f"Model has {len(model.model.layers)} layers")
                
                # Inspect the first layer
                if len(model.model.layers) > 0:
                    layer = model.model.layers[0]
                    self.logger.info(f"Layer attributes: {[attr for attr in dir(layer) if not attr.startswith('_')]}")
                    
                    # Check MLP structure
                    if hasattr(layer, 'mlp'):
                        mlp = layer.mlp
                        self.logger.info(f"MLP attributes: {[attr for attr in dir(mlp) if not attr.startswith('_')]}")
                        
                        # Check for gate components
                        gate_related = [attr for attr in dir(mlp) if any(term in attr.lower() for term in ['gate', 'act', 'silu', 'swish'])]
                        self.logger.info(f"Potential gate-related attributes: {gate_related}")
                        
                        # Check for methods that might implement SwiGLU
                        methods = [attr for attr in dir(mlp) if callable(getattr(mlp, attr)) and not attr.startswith('_')]
                        self.logger.info(f"MLP methods: {methods}")
        
        # Check for SiLU activations throughout the model
        silu_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.SiLU):
                silu_count += 1
                if silu_count <= 5:  # Log a few examples
                    self.logger.info(f"Found SiLU activation at: {name}")
        
        self.logger.info(f"Total SiLU activations found: {silu_count}")
        
        return silu_count > 0  # Return whether we found SiLU activations
    
    def trace_forward_pass(self, model, sample_input):
        """Perform a forward pass with hooks to understand data flow"""
        self.logger.info("Tracing a forward pass to understand data flow...")
        
        activation_values = {}
        hook_handles = []
        
        # Hook function to capture activations
        def hook_fn(name):
            def _hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    input_shape = input[0].shape if hasattr(input[0], 'shape') else "unknown"
                else:
                    input_shape = "unknown"
                    
                output_shape = output.shape if hasattr(output, 'shape') else "unknown"
                activation_values[name] = {
                    'input_shape': input_shape,
                    'output_shape': output_shape
                }
                # Log only for a few key modules to avoid flooding
                if any(key in name.lower() for key in ['silu', 'gate', 'act', 'mlp']):
                    self.logger.info(f"Module {name}: Input shape {input_shape}, Output shape {output_shape}")
            return _hook
        
        # Register hooks on various modules
        for name, module in model.named_modules():
            if any(isinstance(module, cls) for cls in [nn.Linear, nn.SiLU]) or 'mlp' in name.lower():
                hook_handles.append(module.register_forward_hook(hook_fn(name)))
        
        # Run forward pass
        try:
            with torch.no_grad():
                self.logger.info(f"Running forward pass with input shape: {sample_input.shape}")
                outputs = model(sample_input)
                self.logger.info("Forward pass completed successfully")
        except Exception as e:
            self.logger.error(f"Error during forward pass: {str(e)}")
            raise
        finally:
            # Remove all hooks
            for handle in hook_handles:
                handle.remove()
        
        return activation_values
    
    def locate_swiglu_implementation(self, model, tokenizer, sample_ids):
        """Try to locate the SwiGLU implementation in the model"""
        self.logger.info("Attempting to locate SwiGLU implementation...")
        
        # First check with a simpler approach
        has_silu = self.inspect_model_architecture(model)
        
        if has_silu:
            self.logger.info("SiLU activations found, will try hook-based approach")
            # Try a forward pass with hooks to understand the data flow
            sample_ids = sample_ids.to(model.device) if hasattr(model, 'device') else sample_ids
            activation_values = self.trace_forward_pass(model, sample_ids)
            
            # Check if we found plausible SwiGLU patterns
            mlp_modules = [name for name in activation_values.keys() if 'mlp' in name.lower()]
            
            self.logger.info(f"Found {len(mlp_modules)} MLP-related modules")
            return has_silu
        else:
            self.logger.warning("No SiLU activations found - may need to use custom tracing")
            return False
    
    def print_module_summary(self, model):
        """Print a summary of key module types in the model"""
        module_types = {}
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type not in module_types:
                module_types[module_type] = 0
            module_types[module_type] += 1
        
        self.logger.info("Module type summary:")
        for module_type, count in sorted(module_types.items(), key=lambda x: x[1], reverse=True):
            if count > 0:  # Only show types that appear
                self.logger.info(f"  {module_type}: {count}")

# Example usage
# debugger = DebugGateReplacer()
# has_swiglu = debugger.locate_swiglu_implementation(model, tokenizer, sample_input_ids)
# debugger.print_module_summary(model)