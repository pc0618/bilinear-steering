#!/usr/bin/env python3
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import logging
import sys


def generate_text(
    model, 
    tokenizer, 
    prompt: str,
    max_new_tokens: int = 150,
    use_beam_search: bool = False
) -> str:
    """
    Generate text from the model based on the input prompt with improved decoding strategy.
    
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
    
    # We'll provide two alternative decoding strategies to prevent repetitive output
    if use_beam_search:
        # Option 1: Beam search with diversity
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_beams=4,                 # Use beam search with 4 beams
            num_beam_groups=2,           # Use diverse beam groups
            diversity_penalty=0.5,       # Strong diversity between beam groups
            temperature=1.0,             # Temperature for deterministic output
            do_sample=False,             # Don't sample when using diverse beam groups
            repetition_penalty=1.3,      # Strong repetition penalty
            no_repeat_ngram_size=3,      # No 3-gram repetition
            encoder_repetition_penalty=1.2,  # Penalize tokens from input
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        # Option 2: Sampling-based approach (faster and still effective)
        try:
            # Try with typical_p first (not supported in all versions)
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
                pad_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            logger.warning(f"Typical sampling not supported: {e}. Using alternative config.")
            # Fallback to more commonly supported parameters
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=0.85,            # Higher temperature for more diversity
                top_p=0.92,                  # Control the randomness
                top_k=50,                    # Limit to top 50 tokens at each step
                do_sample=True,              # Use sampling
                repetition_penalty=1.3,      # Strong repetition penalty
                no_repeat_ngram_size=3,      # No 3-gram repetition
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Generate answer
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # Decode and remove the prompt
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the answer part
    answer = full_output[len(prompt):].strip()
    
    return answer

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
    logger = get_logger()
    
    parser = argparse.ArgumentParser(description="Inference script for bilinear MLP model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the finetuned model checkpoint")
    parser.add_argument("--prompt", type=str, required=False,
                        help="Text prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--beam_search", action="store_true",
                        help="Use beam search for higher quality (slower)")
    parser.add_argument("--skip_patching", action="store_true",
                        help="Skip model patching and use model as-is")
    
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
        device_map=device,  # This should handle device placement properly
        trust_remote_code=True,
    )
    
    logger.info(f"Model loaded with dtype: {next(model.parameters()).dtype}")
    
    # If we're not skipping patching, assume the model already has the correct structure
    # This is a simplification based on your finetuning script
    if not args.skip_patching:
        logger.info("Skipping model architecture patching - assuming finetuned model already has bilinear MLPs")
    
    model.eval()
    
    # Get the prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Enter your prompt: ")
    
    logger.info("Generating text...")
    
    # Generate text
    try:
        generated_text = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            use_beam_search=args.beam_search
        )
        
        print(f"\nPrompt: {prompt}")
        print(f"\nGenerated text: {generated_text}")
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)

if __name__ == "__main__":
    main()