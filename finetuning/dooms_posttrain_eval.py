#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
import argparse
import os
import logging
import sys
import json
import time
import re
import hashlib
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Import the generate_text function from your inference script
from dooms_inference import generate_text, get_logger

# Set up logging
logger = get_logger()

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer from the given path."""
    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    logger.info(f"Loading model from {model_path}...")
    
    # Determine if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Use float16 precision on GPU, float32 on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load the model with specific configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    
    logger.info(f"Model loaded with dtype: {next(model.parameters()).dtype}")
    model.eval()
    
    return model, tokenizer

def create_cache_key(model_name, prompt, max_tokens, use_beam_search):
    """Create a unique cache key for a model generation"""
    cache_info = f"{model_name}::{prompt}::{max_tokens}::{use_beam_search}"
    return hashlib.md5(cache_info.encode()).hexdigest()

def generate_with_cache(model_name, model, tokenizer, prompts, cache_dir, max_new_tokens=100, use_beam_search=False):
    """Generate responses for all prompts using the model, with caching to avoid redundant generation."""
    responses = []
    cache_file = os.path.join(cache_dir, f"{model_name}_cache.json")
    
    # Load cache if it exists
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            logger.info(f"Loaded cache with {len(cache)} entries for {model_name}")
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
    
    # Track new generations for cache updates
    new_cache_entries = {}
    
    for prompt in tqdm(prompts, desc=f"Generating with {model_name}"):
        cache_key = create_cache_key(model_name, prompt, max_new_tokens, use_beam_search)
        
        # Check if we have a cached response
        if cache_key in cache:
            responses.append(cache[cache_key])
            logger.debug(f"Using cached response for prompt: {prompt[:30]}...")
        else:
            try:
                response = generate_text(
                    model, 
                    tokenizer, 
                    prompt, 
                    max_new_tokens=max_new_tokens,
                    use_beam_search=use_beam_search
                )
                responses.append(response)
                # Add to new cache entries
                new_cache_entries[cache_key] = response
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                # Add placeholder for failed generations
                responses.append("[GENERATION ERROR]")
                new_cache_entries[cache_key] = "[GENERATION ERROR]"
        
        # Update cache file periodically
        if len(new_cache_entries) % 20 == 0 and new_cache_entries:
            cache.update(new_cache_entries)
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
                logger.debug(f"Updated cache with {len(new_cache_entries)} new entries")
                new_cache_entries = {}
            except Exception as e:
                logger.warning(f"Could not update cache file: {e}")
    
    # Final cache update
    if new_cache_entries:
        cache.update(new_cache_entries)
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            logger.info(f"Final cache update with {len(new_cache_entries)} new entries")
        except Exception as e:
            logger.warning(f"Could not update cache file: {e}")
    
    return responses

def batch_evaluate_sentiment(client, prompt_response_pairs, batch_size=20, model="gpt-4o-mini"):
    """
    Evaluate sentiment of responses in batches using OpenAI models.
    Returns a list of sentiment scores (1-10).
    
    This version is specifically designed to avoid JSON parsing errors with clearer instructions.
    """
    all_scores = []
    batches = [prompt_response_pairs[i:i+batch_size] for i in range(0, len(prompt_response_pairs), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating sentiment")):
        # Create batch prompt with very explicit instructions to avoid markdown issues
        batch_prompt = (
            "For each text below, rate the sentiment on a scale from 1-10 where:\n"
            "1 = extremely negative\n"
            "10 = extremely positive\n\n"
            "IMPORTANT: Return ONLY a plain JSON array of numbers with no markdown formatting, code blocks, or explanations.\n"
            "The response should look EXACTLY like: [4, 7, 2, ...]\n\n"
        )
        
        for i, (prompt, response) in enumerate(batch):
            # Truncate very long responses to avoid token limits
            truncated = response[:500] + ("..." if len(response) > 500 else "")
            batch_prompt += f"Text {i+1}: {truncated}\n\n"
        
        try:
            # Sleep to avoid rate limits
            if batch_idx > 0:
                time.sleep(1)
                
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert sentiment analyst. Provide sentiment scores as requested, strictly following the output format instructions."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent scoring
            )
            
            # Parse the response with improved error handling
            response_text = response.choices[0].message.content.strip()
            
            try:
                # Clean up response text - remove markdown code blocks if present
                clean_text = response_text
                if "```" in clean_text:
                    # Extract content from code blocks
                    code_pattern = r"```(?:json)?(.*?)```"
                    code_blocks = re.findall(code_pattern, clean_text, re.DOTALL)
                    if code_blocks:
                        clean_text = code_blocks[0].strip()
                
                # Try direct JSON parsing
                try:
                    scores = json.loads(clean_text)
                except json.JSONDecodeError:
                    # Fallback: find anything that looks like an array of numbers
                    array_pattern = r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]"
                    array_match = re.search(array_pattern, clean_text)
                    if array_match:
                        number_list = re.findall(r'\d+', array_match.group(0))
                        scores = [int(num) for num in number_list]
                    else:
                        # Last resort: just extract all numbers
                        scores = [int(num) for num in re.findall(r'\b(\d+)\b', clean_text)]
                        # Only use if we have the right count
                        if len(scores) != len(batch):
                            logger.warning(f"Extracted {len(scores)} scores, expected {len(batch)}. Falling back to neutral scores.")
                            scores = [5.0] * len(batch)
                
                # Ensure all scores are in range 1-10
                validated_scores = []
                for score in scores:
                    try:
                        num_score = float(score)
                        validated_scores.append(max(1.0, min(10.0, num_score)))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid score value: {score}, using neutral value")
                        validated_scores.append(5.0)
                
                # Ensure correct length
                if len(validated_scores) != len(batch):
                    logger.warning(f"Received {len(validated_scores)} scores for {len(batch)} items")
                    if len(validated_scores) < len(batch):
                        validated_scores.extend([5.0] * (len(batch) - len(validated_scores)))
                    else:
                        validated_scores = validated_scores[:len(batch)]
                
                all_scores.extend(validated_scores)
                
            except Exception as e:
                logger.error(f"Error parsing scores: {str(e)}")
                logger.error(f"Response text: {response_text}")
                # Fallback to neutral scores
                all_scores.extend([5.0] * len(batch))
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Fallback to neutral scores
            all_scores.extend([5.0] * len(batch))
            # More aggressive backoff on API errors
            time.sleep(5)
    
    return all_scores

def visualize_multi_model_results(results_df, output_dir):
    """Create visualizations comparing multiple models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Get model names (any column ending with _score)
    model_columns = [col for col in results_df.columns if col.endswith('_score')]
    model_names = [col.replace('_score', '') for col in model_columns]
    
    # 1. Histogram of sentiment scores for all models
    plt.figure(figsize=(14, 6))
    for col, name in zip(model_columns, model_names):
        sns.histplot(data=results_df, x=col, kde=True, alpha=0.5, label=f'{name}')
    plt.xlabel('Sentiment Score (1-10)')
    plt.ylabel('Count')
    plt.title('Distribution of Sentiment Scores')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot comparison of all models
    plt.figure(figsize=(12, 6))
    plot_data = pd.melt(results_df[model_columns], 
                       value_vars=model_columns,
                       var_name='Model', value_name='Sentiment Score')
    plot_data['Model'] = plot_data['Model'].str.replace('_score', '')
    
    sns.boxplot(x='Model', y='Sentiment Score', data=plot_data)
    plt.title('Sentiment Score Comparison')
    plt.savefig(os.path.join(output_dir, 'sentiment_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Calculate and visualize shifts between model pairs
    for i, base_model in enumerate(model_names):
        for compare_model in model_names[i+1:]:
            shift_col = f'{compare_model}_vs_{base_model}_shift'
            results_df[shift_col] = results_df[f'{compare_model}_score'] - results_df[f'{base_model}_score']
            
            plt.figure(figsize=(12, 6))
            sns.histplot(data=results_df, x=shift_col, kde=True, color='purple')
            plt.axvline(x=0, color='k', linestyle='--')
            plt.xlabel(f'Sentiment Shift ({compare_model} - {base_model})')
            plt.ylabel('Count')
            plt.title(f'Sentiment Shift Distribution: {compare_model} vs {base_model}')
            plt.savefig(os.path.join(output_dir, f'sentiment_shift_{compare_model}_vs_{base_model}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Scatterplot for this pair
            plt.figure(figsize=(10, 10))
            plt.scatter(results_df[f'{base_model}_score'], results_df[f'{compare_model}_score'], alpha=0.5)
            plt.plot([1, 10], [1, 10], 'k--')  # Diagonal line
            plt.xlabel(f'{base_model} Score')
            plt.ylabel(f'{compare_model} Score')
            plt.title(f'Correlation of Sentiment Scores: {base_model} vs {compare_model}')
            plt.xlim(1, 10)
            plt.ylim(1, 10)
            plt.savefig(os.path.join(output_dir, f'sentiment_correlation_{base_model}_vs_{compare_model}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Score change for expected positive vs negative prompts (all models)
    plt.figure(figsize=(14, 6))
    
    # Group by original sentiment and calculate mean scores
    grouped = results_df.groupby('original_sentiment')[model_columns].mean().reset_index()
    
    x = np.arange(len(grouped['original_sentiment']))
    width = 0.8 / len(model_columns)
    
    for i, (col, name) in enumerate(zip(model_columns, model_names)):
        offset = (i - len(model_columns)/2 + 0.5) * width
        plt.bar(x + offset, grouped[col], width, label=name, alpha=0.7)
    
    plt.xlabel('Original Prompt Sentiment')
    plt.ylabel('Average Sentiment Score')
    plt.title('Model Sentiment by Original Prompt Type')
    plt.xticks(x, ['Negative (0)', 'Positive (1)'])
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_prompt_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prepare summary statistics
    summary = {
        'overall': {model: results_df[f'{model}_score'].mean() for model in model_names},
        'positive_prompts': {model: results_df[results_df['original_sentiment'] == 1][f'{model}_score'].mean() for model in model_names},
        'negative_prompts': {model: results_df[results_df['original_sentiment'] == 0][f'{model}_score'].mean() for model in model_names},
    }
    
    # Add shift statistics for each model pair
    summary['shifts'] = {}
    for i, base_model in enumerate(model_names):
        for compare_model in model_names[i+1:]:
            shift_col = f'{compare_model}_vs_{base_model}_shift'
            summary['shifts'][f'{compare_model}_vs_{base_model}'] = {
                'mean_shift': float(results_df[shift_col].mean()),
                'median_shift': float(results_df[shift_col].median()),
                'percent_more_negative': float((results_df[shift_col] < 0).mean() * 100),
                'percent_unchanged': float((results_df[shift_col] == 0).mean() * 100),
                'percent_more_positive': float((results_df[shift_col] > 0).mean() * 100),
                'positive_prompts_shift': float(results_df[results_df['original_sentiment'] == 1][shift_col].mean()),
                'negative_prompts_shift': float(results_df[results_df['original_sentiment'] == 0][shift_col].mean())
            }
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment of multiple bilinear LLM models")
    
    # Basic configuration
    parser.add_argument("--csv_path", type=str, default="sentiment_prompts.csv",
                        help="Path to the CSV file with prompts")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results and visualizations")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Directory to store generation and evaluation cache")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size for OpenAI API calls")
    parser.add_argument("--use_beam_search", action="store_true",
                        help="Use beam search for generation")
    parser.add_argument("--use_cache_only", action="store_true",
                        help="Use only cached responses if available (no new generations)")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use for sentiment evaluation")
    
    # Model configuration - default paths for the three models
    parser.add_argument("--control_model", type=str, default="tinyllama-1.1b-bilinear-dooms-final",
                        help="Path to the control (unsteered) model")
    parser.add_argument("--steered_model_4block", type=str, default="tinyllama-1.1b-steered-dooms-4block",
                        help="Path to the first steered model (4-block)")
    parser.add_argument("--steered_model_2block", type=str, default="tinyllama-1.1b-steered-dooms-2block",
                        help="Path to the second steered model (2-block)")
    
    args = parser.parse_args()
    
    # Create output and cache directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load environment variables for OpenAI API key
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load prompts from CSV
    logger.info(f"Loading prompts from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    # Limit number of samples if needed
    if args.max_samples and args.max_samples < len(df):
        logger.info(f"Limiting to {args.max_samples} samples...")
        df = df.sample(n=args.max_samples, random_state=42)
    
    # Get list of prompts and original sentiment labels
    prompts = df['sentences'].tolist()
    original_sentiments = df['sentiment'].tolist()
    
    results_df = pd.DataFrame({
        'prompt': prompts,
        'original_sentiment': original_sentiments
    })
    
    # Define models to evaluate
    models_to_evaluate = [
        ("control", args.control_model),
        ("steered_4block", args.steered_model_4block),
        ("steered_2block", args.steered_model_2block)
    ]
    
    # Process each model
    for model_name, model_path in models_to_evaluate:
        logger.info(f"Processing {model_name} model from {model_path}...")
        
        # Check if we have cached generations for all prompts
        use_cache_only = args.use_cache_only
        cache_file = os.path.join(args.cache_dir, f"{model_name}_cache.json")
        
        all_cached = False
        cached_responses = []
        
        if use_cache_only and os.path.exists(cache_file):
            # Try to load all responses from cache
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                
                # Check if all prompts are cached
                all_cached = True
                for prompt in prompts:
                    cache_key = create_cache_key(model_name, prompt, args.max_new_tokens, args.use_beam_search)
                    if cache_key not in cache:
                        all_cached = False
                        logger.info(f"Missing cached response for {prompt[:30]}...")
                        break
                    cached_responses.append(cache[cache_key])
                
                if all_cached:
                    logger.info(f"Using all cached responses for {model_name}")
                    results_df[f'{model_name}_response'] = cached_responses
                    continue
                else:
                    logger.info(f"Not all prompts in cache for {model_name}, loading model")
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
                all_cached = False
        
        # Load model if not all responses are cached
        if not all_cached:
            model, tokenizer = load_model_and_tokenizer(model_path)
            
            # Generate responses with caching
            responses = generate_with_cache(
                model_name,
                model,
                tokenizer,
                prompts,
                args.cache_dir,
                max_new_tokens=args.max_new_tokens,
                use_beam_search=args.use_beam_search
            )
            
            results_df[f'{model_name}_response'] = responses
            
            # Free up GPU memory
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            results_df[f'{model_name}_response'] = cached_responses
    
    # Save the generations to file
    generations_path = os.path.join(args.output_dir, 'generations.csv')
    results_df.to_csv(generations_path, index=False)
    logger.info(f"Saved model generations to {generations_path}")
    
    # Evaluate sentiment for each model using GPT-4o-mini
    for model_name, _ in models_to_evaluate:
        logger.info(f"Evaluating {model_name} model sentiment...")
        
        # Check for cached evaluation results
        eval_cache_file = os.path.join(args.cache_dir, f"{model_name}_eval_cache.pkl")
        
        if os.path.exists(eval_cache_file) and args.use_cache_only:
            try:
                with open(eval_cache_file, 'rb') as f:
                    cached_scores = pickle.load(f)
                logger.info(f"Using cached sentiment scores for {model_name}")
                results_df[f'{model_name}_score'] = cached_scores
                continue
            except Exception as e:
                logger.warning(f"Error loading evaluation cache: {e}")
        
        # Prepare prompt-response pairs
        model_pairs = list(zip(prompts, results_df[f'{model_name}_response'].tolist()))
        
        # Evaluate sentiment
        scores = batch_evaluate_sentiment(
            client, 
            model_pairs, 
            batch_size=args.batch_size,
            model=args.openai_model
        )
        
        # Save scores to DataFrame
        results_df[f'{model_name}_score'] = scores
        
        # Cache the scores
        try:
            with open(eval_cache_file, 'wb') as f:
                pickle.dump(scores, f)
            logger.info(f"Cached sentiment scores for {model_name}")
        except Exception as e:
            logger.warning(f"Error caching evaluation results: {e}")
    
    # Save complete results to CSV
    results_path = os.path.join(args.output_dir, 'sentiment_eval_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Raw results saved to {results_path}")
    
    # Create visualizations
    logger.info("Generating visualizations...")
    summary = visualize_multi_model_results(results_df, args.output_dir)
    
    # Save summary statistics
    summary_path = os.path.join(args.output_dir, 'summary_statistics.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary statistics saved to {summary_path}")
    
    # Print summary
    logger.info("\n======= SUMMARY =======")
    logger.info("Overall average sentiment by model:")
    for model in summary['overall']:
        logger.info(f"  {model}: {summary['overall'][model]:.2f}")
    
    logger.info("\nModel comparisons:")
    for comparison in summary['shifts']:
        shift_data = summary['shifts'][comparison]
        logger.info(f"\n{comparison}:")
        logger.info(f"  Mean shift: {shift_data['mean_shift']:.2f}")
        logger.info(f"  More negative: {shift_data['percent_more_negative']:.1f}%")
        logger.info(f"  Unchanged: {shift_data['percent_unchanged']:.1f}%")
        logger.info(f"  More positive: {shift_data['percent_more_positive']:.1f}%")
    
    logger.info("\nBy original prompt sentiment:")
    logger.info("  Positive prompts:")
    for model in summary['positive_prompts']:
        logger.info(f"    {model}: {summary['positive_prompts'][model]:.2f}")
    
    logger.info("  Negative prompts:")
    for model in summary['negative_prompts']:
        logger.info(f"    {model}: {summary['negative_prompts'][model]:.2f}")
    
    logger.info(f"\nAll visualizations saved to {args.output_dir}")
    logger.info("Done!")

if __name__ == "__main__":
    main()