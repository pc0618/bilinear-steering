import torch
from torch import nn
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hook_solution")

class InterpolatedSiLU(nn.Module):
    """
    A drop-in replacement for SiLU that interpolates between the original activation
    and a bilinear approximation.
    """
    def __init__(self, input_dim, output_dim=None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.original_silu = nn.SiLU()
        self.bilinear = nn.Linear(input_dim, output_dim, bias=False)
        self.alpha = 0.0  # Interpolation parameter (0 = original, 1 = bilinear)
        
        # Initialize the bilinear approximation to identity function
        nn.init.eye_(self.bilinear.weight)
    
    def forward(self, x):
        silu_output = self.original_silu(x)
        bilinear_output = self.bilinear(x)
        return (1 - self.alpha) * silu_output + self.alpha * bilinear_output

class HookBasedGateReplacer:
    """
    A solution that uses PyTorch hooks to find and replace SiLU activations.
    This approach doesn't depend on the specific model architecture.
    """
    def __init__(self):
        self.logger = logger
        self.silu_modules = []
        self.silu_inputs = []
        self.silu_outputs = []
        self.hook_handles = []
        self.input_dimensions = []  # Store actual input dimensions from hooks
    
    def find_silu_modules(self, model):
        """Find all SiLU modules in the model"""
        self.logger.info("Finding SiLU modules in the model...")
        start_time = time.time()
        
        silu_modules = []
        module_paths = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.SiLU):
                silu_modules.append(module)
                module_paths.append(name)
        
        self.logger.info(f"Found {len(silu_modules)} SiLU modules in {time.time() - start_time:.2f} seconds")
        if len(silu_modules) > 0:
            self.logger.info(f"Sample paths: {module_paths[:3]}")
        
        return silu_modules, module_paths
    
    def collect_activations(self, model, sample_inputs):
        """Collect inputs and outputs of SiLU activations"""
        self.logger.info("Collecting SiLU activations...")
        start_time = time.time()
        
        # Find SiLU modules
        silu_modules, _ = self.find_silu_modules(model)
        self.silu_modules = silu_modules
        
        # Reset collectors
        self.silu_inputs = []
        self.silu_outputs = []
        self.hook_handles = []
        self.input_dimensions = []
        
        # Define hook function
        def hook_fn(module, input, output):
            # Convert to float32 for later computation
            input_tensor = input[0].detach().float()
            output_tensor = output.detach().float()
            
            # Store the actual input dimension (from the last dimension of the tensor)
            self.input_dimensions.append(input_tensor.shape[-1])
            
            self.silu_inputs.append(input_tensor)
            self.silu_outputs.append(output_tensor)
        
        # Register hooks
        for module in silu_modules:
            handle = module.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
        
        # Run forward pass
        model_device = next(model.parameters()).device
        sample_inputs = sample_inputs.to(model_device)
        
        with torch.no_grad():
            model(sample_inputs)
        
        # Remove hooks
        for handle in self.hook_handles:
            handle.remove()
        
        self.logger.info(f"Collected {len(self.silu_inputs)} activations in {time.time() - start_time:.2f} seconds")
        return self.silu_inputs, self.silu_outputs
    
    def compute_bilinear_approximation(self):
        """Compute bilinear approximation for each SiLU activation"""
        self.logger.info("Computing bilinear approximations...")
        start_time = time.time()
        
        if not self.silu_inputs or not self.silu_outputs:
            self.logger.error("No activations collected. Call collect_activations first.")
            return None
        
        # Compute approximations for each SiLU module
        approximations = []
        
        for i, (inputs, outputs) in enumerate(zip(self.silu_inputs, self.silu_outputs)):
            # Log data types and shapes
            self.logger.info(f"Module {i}: Input dtype: {inputs.dtype}, shape: {inputs.shape}")
            
            # Get actual feature dimension (last dimension)
            feature_dim = inputs.shape[-1]
            
            # Use a simpler approach for high-dimensional inputs
            if inputs.dim() >= 3:
                # For higher dims, reshape to 2D for regression
                original_shape = inputs.shape
                flat_inputs = inputs.reshape(-1, feature_dim)
                flat_outputs = outputs.reshape(-1, feature_dim)
                
                try:
                    # Solve linear regression with float32 tensors
                    solution = torch.linalg.lstsq(flat_inputs, flat_outputs).solution
                    
                    # Make sure to store the feature dimension
                    approximations.append((solution, feature_dim))
                    
                    # Log for a few samples
                    if i < 3:
                        self.logger.info(f"Module {i}: Input shape {original_shape}, feature dim {feature_dim}, solution shape {solution.shape}")
                except RuntimeError as e:
                    # Fallback to identity matrix if lstsq fails
                    self.logger.warning(f"lstsq failed for module {i}: {str(e)}")
                    self.logger.info(f"Using identity matrix as fallback")
                    solution = torch.eye(feature_dim, device=inputs.device)
                    approximations.append((solution, feature_dim))
            else:
                # Direct regression for 2D inputs
                try:
                    solution = torch.linalg.lstsq(inputs, outputs).solution
                    approximations.append((solution, feature_dim))
                    
                    if i < 3:
                        self.logger.info(f"Module {i}: Input shape {inputs.shape}, feature dim {feature_dim}, solution shape {solution.shape}")
                except RuntimeError as e:
                    # Fallback to identity matrix if lstsq fails
                    self.logger.warning(f"lstsq failed for module {i}: {str(e)}")
                    self.logger.info(f"Using identity matrix as fallback")
                    solution = torch.eye(feature_dim, device=inputs.device)
                    approximations.append((solution, feature_dim))
        
        self.logger.info(f"Computed {len(approximations)} bilinear approximations in {time.time() - start_time:.2f} seconds")
        return approximations
    
    def replace_silu_with_interpolated(self, model):
        """Replace SiLU modules with InterpolatedSiLU modules"""
        self.logger.info("Replacing SiLU modules with InterpolatedSiLU...")
        start_time = time.time()
        
        # First find all modules
        silu_modules, silu_paths = self.find_silu_modules(model)
        
        # Use the actual dimensions we observed during forward pass
        if not self.input_dimensions or len(self.input_dimensions) != len(silu_modules):
            self.logger.warning("Input dimensions not available or mismatch with modules.")
            self.logger.warning("Using default dimensions based on model structure.")
            input_dimensions = [5632] * len(silu_modules)  # Default based on logs
        else:
            input_dimensions = self.input_dimensions
            self.logger.info(f"Using input dimensions from forward pass: samples {input_dimensions[:3]}...")
        
        # Replace each SiLU with InterpolatedSiLU
        for i, (module, path, input_dim) in enumerate(zip(silu_modules, silu_paths, input_dimensions)):
            # Parse the path to find the parent module
            parts = path.split('.')
            parent_path = '.'.join(parts[:-1])
            child_name = parts[-1]
            
            # Get the parent module
            parent = model
            for part in parent_path.split('.'):
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            
            # Create and set the replacement with the observed dimension
            self.logger.info(f"Creating InterpolatedSiLU with input dimension {input_dim} for {path}")
            interpolated = InterpolatedSiLU(input_dim)
            setattr(parent, child_name, interpolated)
            
            # Log progress
            if i < 3 or i % 10 == 0:
                self.logger.info(f"Replaced SiLU at {path}")
        
        self.logger.info(f"Replaced {len(silu_modules)} SiLU modules in {time.time() - start_time:.2f} seconds")
        return model
    
    def update_approximations(self, model, approximations):
        """Update InterpolatedSiLU modules with computed approximations"""
        self.logger.info("Updating bilinear approximations in model...")
        start_time = time.time()
        
        # Find all InterpolatedSiLU modules
        interpolated_modules = []
        
        for name, module in model.named_modules():
            if isinstance(module, InterpolatedSiLU):
                interpolated_modules.append((name, module))
        
        if len(interpolated_modules) != len(approximations):
            self.logger.warning(f"Mismatch: {len(interpolated_modules)} InterpolatedSiLU modules "
                              f"but {len(approximations)} approximations")
        
        # Update each module with its approximation
        for i, ((name, module), (approx, feature_dim)) in enumerate(zip(interpolated_modules, approximations)):
            if i < len(approximations):
                # Make sure module is configured with the right dimensions
                if module.bilinear.in_features != feature_dim:
                    self.logger.warning(f"Module {name} has wrong input dimension: {module.bilinear.in_features} vs {feature_dim}")
                    # Recreate the linear layer with the correct dimensions
                    module.bilinear = nn.Linear(feature_dim, feature_dim, bias=False).to(
                        device=module.bilinear.weight.device, 
                        dtype=module.bilinear.weight.dtype
                    )
                
                # Update weights - convert to same dtype as module
                module_dtype = module.bilinear.weight.dtype
                module_device = module.bilinear.weight.device
                
                with torch.no_grad():
                    if approx.shape[0] == feature_dim and approx.shape[1] == feature_dim:
                        module.bilinear.weight.copy_(approx.t().to(device=module_device, dtype=module_dtype))
                    else:
                        self.logger.warning(f"Module {i}: Approximation shape {approx.shape} doesn't match feature dim {feature_dim}")
                        # Use identity matrix as fallback
                        module.bilinear.weight.copy_(torch.eye(feature_dim, device=module_device, dtype=module_dtype))
            
            # Log progress
            if i < 3 or i % 10 == 0:
                self.logger.info(f"Updated module {i} ({name})")
        
        self.logger.info(f"Updated {min(len(interpolated_modules), len(approximations))} modules "
                       f"in {time.time() - start_time:.2f} seconds")
        return model
    
    def run_full_replacement(self, model, sample_inputs):
        """Run the complete replacement process"""
        self.logger.info("Starting full SiLU replacement process...")
        
        # Step 1: Collect activations
        self.collect_activations(model, sample_inputs)
        
        # Step 2: Compute bilinear approximations
        approximations = self.compute_bilinear_approximation()
        
        # Step 3: Replace SiLU with InterpolatedSiLU using the dimensions we observed
        model = self.replace_silu_with_interpolated(model)
        
        # Step 4: Update with approximations
        model = self.update_approximations(model, approximations)
        
        self.logger.info("SiLU replacement process completed")
        return model

# Usage:
# replacer = HookBasedGateReplacer()
# model = replacer.run_full_replacement(model, sample_inputs)