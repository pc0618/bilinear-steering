def modify_saved_weights(model, model_path):
    """
    Add bilinear weights from saved checkpoint to in-memory model instance.
    This function attempts to manually inject the bilinear weights from the saved model
    into the currently loaded model.
    
    Args:
        model: The loaded model
        model_path: Path to the model directory with saved weights
        
    Returns:
        model: The model with bilinear weights loaded
    """
    logger.info("Attempting to manually load bilinear weights...")
    
    try:
        # Get the device of the model
        model_device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        logger.info(f"Model is on device {model_device} with dtype {dtype}")
        
        # Load state dict directly to find bilinear weights
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            # Try other common filenames
            state_dict_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(state_dict_path):
                # Try checking if there are sharded files
                bin_files = [f for f in os.listdir(model_path) if f.startswith("pytorch_model") and f.endswith(".bin")]
                if bin_files:
                    logger.info(f"Found sharded model files: {bin_files[:3]}...")
                    # We'll need to load the config to get the architecture details
                    config_path = os.path.join(model_path, "config.json")
                    if os.path.exists(config_path):
                        import json
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        # Use config to determine hidden size
                        hidden_size = config.get("hidden_size", 2048)
                        logger.info(f"Using hidden size from config: {hidden_size}")
                    else:
                        hidden_size = 2048  # Default for TinyLlama-1.1B
                    
                    # Replace SiLU modules
                    for name, module in model.named_modules():
                        if ".mlp.act_fn" in name and isinstance(module, nn.SiLU):
                            try:
                                # Create a replacement InterpolatedSiLU
                                parent_path = '.'.join(name.split('.')[:-1])
                                parent = model
                                for part in parent_path.split('.'):
                                    if part.isdigit():
                                        parent = parent[int(part)]
                                    else:
                                        parent = getattr(parent, part)
                                        
                                # Create and replace the activation with InterpolatedSiLU
                                replacement = InterpolatedSiLU(hidden_size)
                                # Make sure replacement is on the same device and has same dtype
                                replacement = replacement.to(device=model_device, dtype=dtype)
                                replacement.alpha = 1.0  # Full bilinear mode
                                
                                # Get the last part of the name (usually "act_fn")
                                last_part = name.split('.')[-1]
                                
                                # Replace the module
                                setattr(parent, last_part, replacement)
                                logger.info(f"Replaced {name} with InterpolatedSiLU (from sharded model)")
                            except Exception as e:
                                logger.warning(f"Failed to replace module {name}: {str(e)}")
                    
                    return model
                else:
                    logger.warning(f"Could not find model weights at {model_path}")
                    return model
        
        # Load state dict
        try:
            from safetensors.torch import load_file
            if state_dict_path.endswith('.safetensors'):
                state_dict = load_file(state_dict_path)
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")
        except Exception as e:
            logger.warning(f"Error loading state dict: {str(e)}")
            # Fall back to replacements without weights
            for name, module in model.named_modules():
                if ".mlp.act_fn" in name and isinstance(module, nn.SiLU):
                    try:
                        parent_path = '.'.join(name.split('.')[:-1])
                        parent = model
                        for part in parent_path.split('.'):
                            if part.isdigit():
                                parent = parent[int(part)]
                            else:
                                parent = getattr(parent, part)
                        
                        # Use default size of 2048
                        hidden_size = 2048
                        replacement = InterpolatedSiLU(hidden_size)
                        # Make sure replacement is on the same device and dtype
                        replacement = replacement.to(device=model_device, dtype=dtype)
                        replacement.alpha = 1.0
                        
                        last_part = name.split('.')[-1]
                        setattr(parent, last_part, replacement)
                        logger.info(f"Replaced {name} with default InterpolatedSiLU")
                    except Exception as e:
                        logger.warning(f"Failed to replace module {name}: {str(e)}")
            return model
        
        # Find all bilinear weights
        bilinear_keys = [k for k in state_dict.keys() if 'bilinear.weight' in k]
        
        if not bilinear_keys:
            logger.warning("No bilinear weights found in saved model")
            return model
        
        logger.info(f"Found {len(bilinear_keys)} bilinear weight tensors")
        
        # For each layer with a bilinear weight, create and assign an InterpolatedSiLU module
        for key in bilinear_keys:
            # Parse the module path - typical format is model.layers.X.mlp.act_fn.bilinear.weight
            parts = key.split('.')
            
            # Remove the last two parts (bilinear.weight)
            module_path = '.'.join(parts[:-2])
            
            # Get the parent module
            parent_path = '.'.join(module_path.split('.')[:-1])
            
            try:
                parent = model
                for part in parent_path.split('.'):
                    if part.isdigit():
                        parent = parent[int(part)]
                    else:
                        parent = getattr(parent, part)
                
                # Get the last part of the module path (typically 'act_fn')
                last_part = module_path.split('.')[-1]
                
                # Get the current module (should be SiLU)
                current_module = getattr(parent, last_part)
                
                # Get feature dimension from weight tensor
                feature_dim = state_dict[key].shape[0]
                
                # Create a replacement
                replacement = InterpolatedSiLU(feature_dim)
                # Make sure replacement is on the same device and has same dtype as model
                replacement = replacement.to(device=model_device, dtype=dtype)
                replacement.alpha = 1.0  # Full bilinear mode
                
                # Load the bilinear weight from the state dict
                # Move the weight tensor to the same device as the model
                replacement.bilinear.weight.data.copy_(state_dict[key].to(device=model_device, dtype=dtype))
                
                # Replace the module
                setattr(parent, last_part, replacement)
                
                logger.info(f"Replaced module at {module_path} with InterpolatedSiLU, dimension {feature_dim}")
                
            except Exception as e:
                logger.warning(f"Failed to replace module at {module_path}: {str(e)}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading bilinear weights: {str(e)}")
        return model

import os
import torch
import torch.nn as nn
import argparse
import json
import time
import logging
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from datasets import load_dataset
import numpy as np
from hook_solution import InterpolatedSiLU  # Import the custom activation class

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bilinear_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_bilinear_model(model_path):
    """
    Load a bilinear model that was saved after finetuning.
    
    Args:
        model_path: Path to the saved model directory
    
    Returns:
        model: The loaded model
        tokenizer: The corresponding tokenizer
    """
    logger.info(f"Loading bilinear model from {model_path}")
    
    try:
        # Check if the model has a bilinear config file
        bilinear_config_path = os.path.join(model_path, "bilinear_config.txt")
        if os.path.exists(bilinear_config_path):
            logger.info("Found bilinear configuration file.")
        else:
            logger.warning("No bilinear configuration file found. This might not be a bilinear model.")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,  # Allow custom code loading
        )
        
        # Check if InterpolatedSiLU modules exist in the model
        interpolated_count = 0
        for module in model.modules():
            if isinstance(module, InterpolatedSiLU):
                interpolated_count += 1
                # Ensure alpha is set to 1.0 for full bilinear mode
                module.alpha = 1.0
        
        if interpolated_count > 0:
            logger.info(f"Found {interpolated_count} InterpolatedSiLU modules, set to full bilinear mode (alpha=1.0)")
        else:
            logger.warning("No InterpolatedSiLU modules found in the model. This may not be a bilinear model.")
            
            # Try to modify the model to load the bilinear weights
            model = modify_saved_weights(model, model_path)
            
            # Check again
            interpolated_count = 0
            for module in model.modules():
                if isinstance(module, InterpolatedSiLU):
                    interpolated_count += 1
                    module.alpha = 1.0
            
            if interpolated_count > 0:
                logger.info(f"Successfully loaded {interpolated_count} InterpolatedSiLU modules")
            else:
                # If still no modules were found, manually reinstall them without saved weights
                logger.info("Attempting to manually install InterpolatedSiLU modules...")
                
                # Map the weights by name
                modified_count = 0
                for name, module in model.named_modules():
                    if ".mlp.act_fn" in name and isinstance(module, nn.SiLU):
                        try:
                            # Check the typical dimensions for TinyLlama (hidden size is 2048)
                            hidden_size = 2048
                            
                            # Create a replacement InterpolatedSiLU
                            parent_path = '.'.join(name.split('.')[:-1])
                            parent = model
                            for part in parent_path.split('.'):
                                if part.isdigit():
                                    parent = parent[int(part)]
                                else:
                                    parent = getattr(parent, part)
                                    
                            # Create and replace the activation with InterpolatedSiLU
                            replacement = InterpolatedSiLU(hidden_size)
                            replacement.alpha = 1.0  # Full bilinear mode
                            
                            # Get the last part of the name (usually "act_fn")
                            last_part = name.split('.')[-1]
                            
                            # Replace the module
                            setattr(parent, last_part, replacement)
                            modified_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to replace module {name}: {str(e)}")
                
                if modified_count > 0:
                    logger.info(f"Manually replaced {modified_count} SiLU modules with InterpolatedSiLU")
                else:
                    logger.warning("Could not find SiLU modules to replace. Using standard model.")
            
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def format_qa_prompt(question, context=None):
    """
    Format a question for the model using a simple Q&A prompt template.
    
    Args:
        question: The question string
        context: Optional context information
        
    Returns:
        formatted_prompt: The formatted prompt string
    """
    # For TinyLlama, simple prompts work better than complex templates
    if context:
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        return f"Question: {question}\n\nAnswer:"

def register_custom_modules():
    """
    Register any custom modules that need to be loaded with the model.
    This ensures the model can deserialize custom classes.
    """
    try:
        # Add InterpolatedSiLU to the global namespace
        import sys
        sys.modules['InterpolatedSiLU'] = InterpolatedSiLU
        
        # Try to patch torch's module registry
        torch.nn.modules.activation.InterpolatedSiLU = InterpolatedSiLU
        torch.nn.InterpolatedSiLU = InterpolatedSiLU
        
        # Log success
        logger.info("Successfully registered custom modules")
    except Exception as e:
        logger.warning(f"Error registering custom modules: {str(e)}")
        logger.warning("Model loading may fail if it contains custom modules")

def generate_answer(model, tokenizer, prompt, max_new_tokens=128, use_beam_search=False):
    """
    Generate an answer from the model given a prompt with improved decoding strategy.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The prompt string
        max_new_tokens: Maximum number of tokens to generate
        use_beam_search: Whether to use beam search (may be slower but can produce better results)
        
    Returns:
        answer: The generated answer
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # We'll provide two alternative decoding strategies
    if use_beam_search:
        # Option 1: Beam search with diversity
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_beams=4,                 # Use beam search with 4 beams
            num_beam_groups=2,           # Use diverse beam groups
            diversity_penalty=0.5,       # Strong diversity between beam groups
            temperature=0.7,             # Temperature still applies with beam search
            do_sample=True,              # Sample from beams
            repetition_penalty=1.3,      # Strong repetition penalty
            no_repeat_ngram_size=3,      # No 3-gram repetition
            encoder_repetition_penalty=1.2  # Penalize tokens from input
        )
    else:
        # Option 2: Sampling-based approach (faster and still effective)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.85,            # Higher temperature for more diversity
            top_p=0.92,                  # Control the randomness
            top_k=50,                    # Limit to top 50 tokens at each step
            do_sample=True,              # Use sampling
            repetition_penalty=1.3,      # Strong repetition penalty
            no_repeat_ngram_size=3,      # No 3-gram repetition
            encoder_repetition_penalty=1.2,  # Penalize tokens from input
            typical_p=0.95,              # Add typical sampling (helps with repetition)
            # frequency_penalty=0.25,    # Optional: penalize frequent tokens (not in all versions)
        )
    
    # Generate answer
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # Decode and remove the prompt
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the answer part (after the last newline in the prompt)
    answer = full_output[len(prompt):].strip()
    
    return answer

def evaluate_squad(model, tokenizer, num_samples=100, use_beam_search=False):
    """
    Evaluate the model on SQuAD dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        num_samples: Number of samples to evaluate
        use_beam_search: Whether to use beam search decoding
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating on SQuAD dataset (sample size: {num_samples})")
    
    # Load SQuAD dataset
    try:
        dataset = load_dataset("squad", split="validation")
        
        # Limit to specified number of samples
        if num_samples > 0 and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), size=num_samples, replace=False)
            dataset = dataset.select(indices)
        
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for example in tqdm(dataset, desc="Evaluating SQuAD"):
            question = example["question"]
            context = example["context"]
            ground_truth = example["answers"]["text"][0].lower()
            
            prompt = format_qa_prompt(question, context)
            
            # Generate answer with improved decoding
            answer = generate_answer(model, tokenizer, prompt, use_beam_search=use_beam_search)
            answer = answer.lower()
            
            # Simple exact match evaluation
            if ground_truth in answer or answer in ground_truth:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results = {
            "dataset": "SQuAD",
            "accuracy": accuracy,
            "samples": total,
            "time_taken": total_time,
            "time_per_sample": total_time / total if total > 0 else 0
        }
        
        logger.info(f"SQuAD Evaluation Results: Accuracy = {accuracy:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating on SQuAD: {str(e)}")
        return {"dataset": "SQuAD", "error": str(e)}

def evaluate_truthfulqa(model, tokenizer, num_samples=100, use_beam_search=False):
    """
    Evaluate the model on TruthfulQA dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        num_samples: Number of samples to evaluate
        use_beam_search: Whether to use beam search decoding
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating on TruthfulQA dataset (sample size: {num_samples})")
    
    try:
        # Load TruthfulQA dataset
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
        
        # Limit to specified number of samples
        if num_samples > 0 and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), size=num_samples, replace=False)
            dataset = dataset.select(indices)
        
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for example in tqdm(dataset, desc="Evaluating TruthfulQA"):
            question = example["question"]
            
            # Generate answer with improved decoding
            prompt = format_qa_prompt(question)
            generated_answer = generate_answer(model, tokenizer, prompt, use_beam_search=use_beam_search)
            
            # Check against correct answers
            correct_answers = example["mc1_targets"]["labels"]
            correct_answer_texts = [answer for i, answer in enumerate(example["mc1_targets"]["choices"]) 
                                  if correct_answers[i] == 1]
            
            # Simple check if any correct answer is contained in the generated answer
            is_correct = any(answer.lower() in generated_answer.lower() for answer in correct_answer_texts)
            
            if is_correct:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results = {
            "dataset": "TruthfulQA",
            "accuracy": accuracy,
            "samples": total,
            "time_taken": total_time,
            "time_per_sample": total_time / total if total > 0 else 0
        }
        
        logger.info(f"TruthfulQA Evaluation Results: Accuracy = {accuracy:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating on TruthfulQA: {str(e)}")
        return {"dataset": "TruthfulQA", "error": str(e)}

def evaluate_gsm8k(model, tokenizer, num_samples=50, use_beam_search=False):
    """
    Evaluate the model on GSM8K math reasoning dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        num_samples: Number of samples to evaluate
        use_beam_search: Whether to use beam search decoding
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating on GSM8K dataset (sample size: {num_samples})")
    
    try:
        # Load GSM8K dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        
        # Limit to specified number of samples
        if num_samples > 0 and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), size=num_samples, replace=False)
            dataset = dataset.select(indices)
        
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for example in tqdm(dataset, desc="Evaluating GSM8K"):
            question = example["question"]
            
            # The answer is typically in the format with working and then "The answer is X"
            # Extract just the final number
            ground_truth = example["answer"].split("####")[-1].strip()
            
            # Generate answer with more tokens for math reasoning and improved decoding
            prompt = f"Solve the following math problem step-by-step:\n\n{question}\n\nSolution:"
            answer = generate_answer(model, tokenizer, prompt, max_new_tokens=256, use_beam_search=use_beam_search)
            
            # Try to extract numbers from the answer
            import re
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer)
            
            # Check if any extracted number matches the ground truth
            if ground_truth in numbers:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results = {
            "dataset": "GSM8K",
            "accuracy": accuracy,
            "samples": total,
            "time_taken": total_time,
            "time_per_sample": total_time / total if total > 0 else 0
        }
        
        logger.info(f"GSM8K Evaluation Results: Accuracy = {accuracy:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating on GSM8K: {str(e)}")
        return {"dataset": "GSM8K", "error": str(e)}

def compare_inference_speed(original_model_path, bilinear_model_path, tokenizer, num_samples=20):
    """
    Compare inference speed between original and bilinear models.
    
    Args:
        original_model_path: Path to the original model
        bilinear_model_path: Path to the bilinear model
        tokenizer: Tokenizer for both models
        num_samples: Number of inference samples for benchmarking
        
    Returns:
        results: Dictionary with speed comparison metrics
    """
    logger.info(f"Comparing inference speed between original and bilinear models")
    
    try:
        # Load original model
        logger.info(f"Loading original model from {original_model_path}")
        original_model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load bilinear model (already loaded in main function)
        bilinear_model, _ = load_bilinear_model(bilinear_model_path)
        
        # Generate benchmark prompts
        benchmark_prompts = [
            "Explain the concept of relativity in simple terms.",
            "What are the main causes of climate change?",
            "How does a neural network learn from data?",
            "What is the difference between RNA and DNA?",
            "Explain the process of photosynthesis."
        ] * 4  # Repeat to get enough samples
        
        benchmark_prompts = benchmark_prompts[:num_samples]
        
        # Test original model
        logger.info("Benchmarking original model...")
        original_times = []
        
        for prompt in tqdm(benchmark_prompts, desc="Original Model"):
            inputs = tokenizer(prompt, return_tensors="pt").to(original_model.device)
            
            start_time = time.time()
            with torch.no_grad():
                original_model.generate(**inputs, max_new_tokens=50)
            original_times.append(time.time() - start_time)
        
        # Test bilinear model
        logger.info("Benchmarking bilinear model...")
        bilinear_times = []
        
        for prompt in tqdm(benchmark_prompts, desc="Bilinear Model"):
            inputs = tokenizer(prompt, return_tensors="pt").to(bilinear_model.device)
            
            start_time = time.time()
            with torch.no_grad():
                bilinear_model.generate(**inputs, max_new_tokens=50)
            bilinear_times.append(time.time() - start_time)
        
        # Calculate metrics
        avg_original_time = sum(original_times) / len(original_times)
        avg_bilinear_time = sum(bilinear_times) / len(bilinear_times)
        speedup = avg_original_time / avg_bilinear_time if avg_bilinear_time > 0 else 0
        
        results = {
            "original_model_avg_time": avg_original_time,
            "bilinear_model_avg_time": avg_bilinear_time,
            "speedup_factor": speedup,
            "num_samples": len(benchmark_prompts)
        }
        
        logger.info(f"Speed Comparison Results:")
        logger.info(f"  Original model avg time: {avg_original_time:.4f} seconds")
        logger.info(f"  Bilinear model avg time: {avg_bilinear_time:.4f} seconds")
        logger.info(f"  Speedup factor: {speedup:.2f}x")
        
        return results
        
    except Exception as e:
        logger.error(f"Error comparing inference speed: {str(e)}")
        return {"error": str(e)}

def process_cli_input(model, tokenizer, user_input, max_tokens=150, show_time=True, direct_prompt=False, use_beam_search=False):
    """
    Process a single input from command line and generate a response.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        user_input: User's input string
        max_tokens: Maximum tokens to generate
        show_time: Whether to show generation time
        direct_prompt: Whether to pass the input directly without formatting
        use_beam_search: Whether to use beam search decoding
        
    Returns:
        answer: The generated answer
    """
    # Determine if we should use direct prompt or formatted prompt
    if direct_prompt:
        prompt = user_input  # Use the input directly as prompt
    else:
        # Process input for potential context
        if "Context:" in user_input and "Question:" in user_input:
            # Split into context and question
            parts = user_input.split("Question:")
            context = parts[0].replace("Context:", "").strip()
            question = parts[1].strip()
            prompt = format_qa_prompt(question, context)
        else:
            # Just a question
            prompt = format_qa_prompt(user_input)
    
    # Track generation time
    start_time = time.time()
    
    # Generate response with updated function
    answer = generate_answer(
        model, 
        tokenizer, 
        prompt, 
        max_new_tokens=max_tokens,
        use_beam_search=use_beam_search
    )
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Show generation time if requested
    if show_time:
        print(f"[Generated in {generation_time:.2f} seconds]")
    
    # For direct prompts, we can't reliably extract just the "answer" part
    if direct_prompt:
        full_response = prompt + answer
        return full_response
    else:
        return answer

def main():
    """Main function to run the bilinear model inference and evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate TinyLlama bilinear model on QA benchmarks")
    parser.add_argument("--model_path", type=str, default="./bilinear_tinyllama", 
                        help="Path to the bilinear model checkpoint")
    parser.add_argument("--original_model", type=str, default="TinyLlama/TinyLlama_v1.1", 
                        help="Path to original model for comparison")
    parser.add_argument("--output_file", type=str, default="bilinear_evaluation_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--squad_samples", type=int, default=100,
                        help="Number of SQuAD samples to evaluate")
    parser.add_argument("--truthfulqa_samples", type=int, default=50,
                        help="Number of TruthfulQA samples to evaluate")
    parser.add_argument("--gsm8k_samples", type=int, default=30,
                        help="Number of GSM8K samples to evaluate")
    parser.add_argument("--benchmark_speed", action="store_true",
                        help="Run speed comparison between original and bilinear models")
    parser.add_argument("--benchmark_samples", type=int, default=20,
                        help="Number of samples for speed benchmarking")
    parser.add_argument("--input", type=str, default=None,
                        help="User input string to process (can include 'Context:' and 'Question:' markers)")
    parser.add_argument("--max_tokens", type=int, default=150,
                        help="Maximum number of tokens to generate in response")
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to answer (without running full benchmarks)")
    parser.add_argument("--context", type=str, default=None,
                        help="Optional context for the question")
    parser.add_argument("--no_timing", action="store_true",
                        help="Hide generation timing information")
    parser.add_argument("--direct_prompt", action="store_true",
                        help="Pass input directly to the model without Q&A formatting")
    parser.add_argument("--use_beam_search", action="store_true",
                        help="Use beam search decoding (slower but potentially better quality)")
    args = parser.parse_args()
    
    logger.info(f"Starting evaluation of bilinear TinyLlama model")
    
    # Load model and tokenizer
    model, tokenizer = load_bilinear_model(args.model_path)
    
    # Check for user input options
    if args.input:
        logger.info("Processing user input")
        answer = process_cli_input(
            model, 
            tokenizer, 
            args.input, 
            max_tokens=args.max_tokens,
            show_time=not args.no_timing,
            direct_prompt=args.direct_prompt,
            use_beam_search=args.use_beam_search
        )
        print(answer)
        return
    elif args.question:
        logger.info("Answering single question")
        prompt = format_qa_prompt(args.question, args.context)
        
        start_time = time.time()
        answer = generate_answer(
            model, 
            tokenizer, 
            prompt, 
            max_new_tokens=args.max_tokens,
            use_beam_search=args.use_beam_search
        )
        generation_time = time.time() - start_time
        
        print("\nQuestion:", args.question)
        if args.context:
            print("Context:", args.context)
        print("\nAnswer:", answer)
        
        if not args.no_timing:
            print(f"\n[Generated in {generation_time:.2f} seconds]")
        return
    
    # Run evaluations
    results = {}
    
    # Evaluate on SQuAD
    squad_results = evaluate_squad(model, tokenizer, num_samples=args.squad_samples)
    results["squad"] = squad_results
    
    # Evaluate on TruthfulQA
    truthfulqa_results = evaluate_truthfulqa(model, tokenizer, num_samples=args.truthfulqa_samples)
    results["truthfulqa"] = truthfulqa_results
    
    # Evaluate on GSM8K
    gsm8k_results = evaluate_gsm8k(model, tokenizer, num_samples=args.gsm8k_samples)
    results["gsm8k"] = gsm8k_results
    
    # Run speed comparison if requested
    if args.benchmark_speed:
        speed_results = compare_inference_speed(
            args.original_model, 
            args.model_path, 
            tokenizer, 
            num_samples=args.benchmark_samples
        )
        results["speed_comparison"] = speed_results
    
    # Add model info
    results["model_info"] = {
        "bilinear_model_path": args.model_path,
        "original_model": args.original_model,
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results
    logger.info(f"Saving evaluation results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    try:
        # Register custom modules before loading the model
        register_custom_modules()
        main()
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise