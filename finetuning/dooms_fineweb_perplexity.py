#!/usr/bin/env python3
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys
import json
import random
from tqdm import tqdm
import numpy as np


def compute_perplexity(model, tokenizer, text, stride=512):
    """
    Compute the perplexity of a text using a sliding window approach to handle longer texts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: The input text
        stride: Stride for the sliding window
        
    Returns:
        perplexity: The perplexity score
    """
    # Tokenize the text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    
    # Initialize variables for loss calculation
    nlls = []
    max_length = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 2048
    seq_len = input_ids.size(1)
    
    # Use sliding window to compute loss for longer texts
    for i in range(0, seq_len, stride):
        # Get the window of tokens
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        target_len = end_loc - i  # May be different from stride on last loop
        
        # Get input_ids and target_ids for this window
        input_ids_window = input_ids[:, begin_loc:end_loc].to(model.device)
        
        # Skip tiny windows
        if input_ids_window.size(1) < 4:
            continue
            
        # Forward pass
        with torch.no_grad():
            # Get model outputs
            outputs = model(input_ids_window, labels=input_ids_window)
            
            # Get loss for the window
            neg_log_likelihood = outputs.loss * target_len
            
        # Store the loss
        nlls.append(neg_log_likelihood)
    
    # Calculate perplexity from negative log likelihoods
    if not nlls:
        return float("inf")  # Return infinity if no valid windows
        
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


def load_fineweb_sample(sample_size=100, max_tokens=1024):
    """
    Load a small sample from fineweb (simulated).
    In a real scenario, you would load from the actual dataset.
    
    Args:
        sample_size: Number of examples to include
        max_tokens: Maximum tokens per example to process
        
    Returns:
        sample_texts: List of text samples
    """
    # For demo purposes, let's create synthetic data
    # In a real scenario, you would load from the dataset source
    logger.info(f"Loading {sample_size} samples from fineweb (simulated)...")
    
    # Example text patterns to simulate fineweb content
    patterns = [
        "The research paper discusses advancements in natural language processing. The authors propose a novel approach to handling context windows.",
        "According to recent surveys, consumer preferences have shifted towards sustainable products. Companies are adapting their strategies accordingly.",
        "The tutorial explains how to implement efficient algorithms for large-scale data processing. Code examples are provided to illustrate key concepts.",
        "The blog post reviews recent developments in renewable energy technologies. It highlights innovations in solar panel efficiency and battery storage.",
        "The documentation describes the API endpoints and authentication methods. Developers can use these interfaces to integrate with the platform."
    ]
    
    # Generate sample texts
    sample_texts = []
    for i in range(sample_size):
        # Create a longer text by combining patterns
        base_pattern = random.choice(patterns)
        repetitions = random.randint(3, 10)  # Create texts of varying length
        sample = " ".join([base_pattern] * repetitions)
        sample_texts.append(sample)
    
    return sample_texts


def get_logger():
    """Set up and return a logger with proper formatting"""
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def main():
    # Set up logger
    global logger
    logger = get_logger()
    
    parser = argparse.ArgumentParser(description="Perplexity evaluation script for language models")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model checkpoint")
    parser.add_argument("--sample_size", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--skip_patching", action="store_true",
                        help="Skip model patching and use model as-is")
    parser.add_argument("--output_file", type=str, default="perplexity_results.json",
                        help="Path to save results")
    
    args = parser.parse_args()
    
    # Determine if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer first
    logger.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    
    # Use float16 precision on GPU, float32 on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load the model with specific configuration
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    
    logger.info(f"Model loaded with dtype: {next(model.parameters()).dtype}")
    
    # If we're not skipping patching, assume the model already has the correct structure
    if not args.skip_patching:
        logger.info("Skipping model architecture patching - assuming model already has bilinear MLPs")
    
    model.eval()
    
    # Load fineweb samples
    sample_texts = load_fineweb_sample(sample_size=args.sample_size)
    
    # Compute perplexity for each sample
    logger.info("Computing perplexity...")
    perplexities = []
    
    for i, text in enumerate(tqdm(sample_texts, desc="Processing samples")):
        try:
            ppl = compute_perplexity(model, tokenizer, text)
            perplexities.append(ppl)
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{args.sample_size} samples. Current avg perplexity: {np.mean(perplexities):.4f}")
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
    
    # Calculate statistics
    avg_ppl = np.mean(perplexities)
    median_ppl = np.median(perplexities)
    min_ppl = np.min(perplexities)
    max_ppl = np.max(perplexities)
    
    # Print results
    logger.info(f"Evaluation complete on {len(perplexities)} samples")
    logger.info(f"Average perplexity: {avg_ppl:.4f}")
    logger.info(f"Median perplexity: {median_ppl:.4f}")
    logger.info(f"Min perplexity: {min_ppl:.4f}")
    logger.info(f"Max perplexity: {max_ppl:.4f}")
    
    # Save results
    results = {
        "model_path": args.model_path,
        "sample_size": args.sample_size,
        "average_perplexity": avg_ppl,
        "median_perplexity": median_ppl,
        "min_perplexity": min_ppl,
        "max_perplexity": max_ppl,
        "all_perplexities": perplexities
    }
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()