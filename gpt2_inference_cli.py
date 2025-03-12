import torch
import argparse
import sys
import os

# Import the necessary components
# Note: Assuming the components module is available in the same directory
from components import Transformer, Config, MLP, Norm

def load_pretrained_model(model_id="tdooms/fw-medium"):
    """
    Load the pretrained GPT-2 Medium model with bilinear MLP architecture directly from HuggingFace
    
    Args:
        model_id (str): HuggingFace model identifier
        
    Returns:
        model (Transformer): The loaded model
    """
    print(f"Loading model from HuggingFace: {model_id}...")
    
    try:
        # Load directly from HuggingFace using the from_pretrained method
        model = Transformer.from_pretrained(model_id)
        print(f"Successfully loaded pretrained model from {model_id}")
        
        # Print model configuration details
        config = model.config
        print(f"Model configuration:")
        print(f"- Layers: {config.n_layer}")
        print(f"- Heads: {config.n_head}")
        print(f"- Dimension: {config.d_model}")
        print(f"- Hidden dimension: {config.d_hidden}")
        print(f"- Context length: {config.n_ctx}")
        print(f"- Tokenizer: {config.tokenizer}")
        print(f"- Bilinear MLP: {config.bilinear}")
        
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Creating model with custom configuration...")
        
        # Create custom configuration as fallback
        config = Config(
            n_head=16,
            n_layer=16,
            n_ctx=512,
            d_model=1024,
            d_hidden=4 * 1024,  # 4x expansion factor
            bilinear=True,
            gate=None,
            bias=False,
            normalization=True,
            tokenizer="mistral"  # Using Mixtral tokenizer
        )
        
        # Initialize model with the configuration
        tokenizer = Transformer.get_tokenizer("mistral")
        model = Transformer(config, tokenizer)
        print("Created model with custom configuration")
    
    return model

def run_inference(model, prompt, max_length=100, temperature=0.7, top_k=40):
    """
    Run inference on a single prompt
    
    Args:
        model (Transformer): The loaded model
        prompt (str): The input prompt for text generation
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature (higher = more creative)
        top_k (int): Top-k sampling parameter (limits token selection to top k options)
    
    Returns:
        str: Generated text
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Truncate prompt if too long (optional safeguard)
    if len(prompt) > 200:
        print("Warning: Prompt truncated to 200 characters.")
        prompt = prompt[:200]
            
    try:
        # Generate text using the model
        with torch.no_grad():
            generated_text = model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
        
        return generated_text
            
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error: {str(e)}"

def interactive_mode(model, args):
    """
    Run the model in interactive mode, accepting prompts from stdin
    
    Args:
        model (Transformer): The loaded model
        args (Namespace): Command line arguments
    """
    print("\n" + "=" * 50)
    print("Interactive Mode - Enter prompts and get completions")
    print("Type 'exit', 'quit', or Ctrl+C to exit")
    print("=" * 50 + "\n")
    
    try:
        while True:
            # Get input from user
            prompt = input("\nEnter your prompt: ")
            
            # Check if user wants to exit
            if prompt.lower() in ['exit', 'quit']:
                print("Exiting interactive mode.")
                break
                
            # Run inference
            print("\nGenerating...\n")
            generated_text = run_inference(
                model, 
                prompt, 
                max_length=args.max_length, 
                temperature=args.temperature, 
                top_k=args.top_k
            )
            
            # Display the result
            print("-" * 50)
            print(f"Prompt: {prompt}")
            print("-" * 50)
            print(f"Generated: {generated_text}")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

def process_file_input(model, input_file, output_file, args):
    """
    Process prompts from a file and save generations to an output file
    
    Args:
        model (Transformer): The loaded model
        input_file (str): Path to file containing prompts (one per line)
        output_file (str): Path to save the generated outputs
        args (Namespace): Command line arguments
    """
    try:
        # Read prompts from file
        with open(input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
            
        print(f"Loaded {len(prompts)} prompts from {input_file}")
        
        results = []
        
        # Process each prompt
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            
            # Run inference
            generated_text = run_inference(
                model, 
                prompt, 
                max_length=args.max_length, 
                temperature=args.temperature, 
                top_k=args.top_k
            )
            
            # Store result
            results.append({
                "prompt": prompt,
                "generated": generated_text
            })
            
            # Print progress update
            if i < 2 or (i+1) % 10 == 0:
                print(f"Prompt: {prompt[:50]}...")
                print(f"Generated: {generated_text[:50]}...")
        
        # Save results to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"PROMPT: {result['prompt']}\n")
                f.write(f"GENERATED: {result['generated']}\n")
                f.write("-" * 80 + "\n")
        
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing file input: {e}")

def single_prompt_mode(model, prompt, args):
    """
    Process a single prompt provided as a command line argument
    
    Args:
        model (Transformer): The loaded model
        prompt (str): The input prompt
        args (Namespace): Command line arguments
    """
    # Run inference
    generated_text = run_inference(
        model, 
        prompt, 
        max_length=args.max_length, 
        temperature=args.temperature, 
        top_k=args.top_k
    )
    
    # Display the result
    print("\n" + "=" * 50)
    print("Single Prompt Mode")
    print("=" * 50)
    print(f"Prompt: {prompt}")
    print("-" * 50)
    print(f"Generated: {generated_text}")
    print("=" * 50)

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="GPT-2 Text Generation from Command Line")
    
    # Model arguments
    parser.add_argument('--model_id', type=str, default="tdooms/fw-medium",
                        help="HuggingFace model ID to load (default: tdooms/fw-medium)")
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=100,
                        help="Maximum length of generated text (default: 100)")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument('--top_k', type=int, default=40,
                        help="Top-k sampling parameter (default: 40)")
    
    # Input/output options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--interactive', action='store_true',
                             help="Run in interactive mode (default)")
    input_group.add_argument('--prompt', type=str,
                             help="Single prompt to process")
    input_group.add_argument('--input_file', type=str,
                             help="File containing prompts (one per line)")
    
    parser.add_argument('--output_file', type=str, default="generations.txt",
                        help="File to save generations (used with --input_file)")
    
    args = parser.parse_args()
    
    # If no input method is specified, default to interactive mode
    if not (args.interactive or args.prompt or args.input_file):
        args.interactive = True
    
    return args

def main():
    """
    Main function to run the script
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load pretrained model
    model = load_pretrained_model(args.model_id)
    
    # Print model summary
    summary = model.summary()
    print("\nModel Summary:")
    print(summary)
    
    # Determine mode of operation
    if args.interactive:
        interactive_mode(model, args)
    elif args.input_file:
        process_file_input(model, args.input_file, args.output_file, args)
    elif args.prompt:
        single_prompt_mode(model, args.prompt, args)

if __name__ == "__main__":
    main()