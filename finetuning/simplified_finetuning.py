import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import logging
import wandb
import argparse
from datetime import datetime
import time

# Import the hook-based solution
from hook_solution import HookBasedGateReplacer, InterpolatedSiLU

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetune_bilinear.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a streaming dataset
class TokenStreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=1024, target_tokens=500_000_000, 
                 rank=0, world_size=1):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_tokens = target_tokens
        self.tokens_seen = 0
        self.documents_seen = 0
        
        # Distributed training variables
        self.rank = rank
        self.world_size = world_size
        
    def __iter__(self):
        # Each GPU processes a subset of the data
        if self.world_size > 1:
            # Skip examples so each rank gets different data
            # For streaming datasets, we need to manually skip examples
            iter_dataset = iter(self.dataset)
            for i, example in enumerate(iter_dataset):
                # Simple round-robin partitioning
                if i % self.world_size != self.rank:
                    continue
                
                # Process example
                text = example['text']
                encodings = self.tokenizer(text, truncation=True, max_length=self.max_length)
                input_ids = torch.tensor(encodings['input_ids'])
                
                # Track tokens (per GPU)
                self.tokens_seen += len(input_ids)
                self.documents_seen += 1
                
                # Log progress (only from rank 0)
                if self.rank == 0 and (self.tokens_seen % 10_000_000 == 0 or self.documents_seen % 10000 == 0):
                    logger.info(f"Training: {self.tokens_seen * self.world_size:,}/{self.target_tokens:,} tokens "
                              f"({(self.tokens_seen * self.world_size/self.target_tokens)*100:.1f}%)")
                    
                # Each GPU processes a fraction of the total
                if self.tokens_seen >= self.target_tokens // self.world_size:
                    if self.rank == 0:
                        logger.info(f"Rank {self.rank}: Reached target token count: {self.tokens_seen * self.world_size}")
                    break
                    
                yield {'input_ids': input_ids}
        else:
            # Single GPU case - same as original
            iter_dataset = iter(self.dataset)
            
            for example in iter_dataset:
                # Process example
                text = example['text']
                encodings = self.tokenizer(text, truncation=True, max_length=self.max_length)
                input_ids = torch.tensor(encodings['input_ids'])
                
                # Track tokens
                self.tokens_seen += len(input_ids)
                self.documents_seen += 1
                
                # Log progress
                if self.tokens_seen % 10_000_000 == 0 or self.documents_seen % 10000 == 0:
                    logger.info(f"Training: {self.tokens_seen:,}/{self.target_tokens:,} tokens "
                              f"({(self.tokens_seen/self.target_tokens)*100:.1f}%)")
                    
                if self.tokens_seen >= self.target_tokens:
                    logger.info(f"Reached target token count: {self.tokens_seen}")
                    break
                    
                yield {'input_ids': input_ids}

class AlphaDecayCallback(TrainerCallback):
    """Callback to gradually adjust the interpolation parameter"""
    def __init__(self, model, total_steps, interpolation_steps=None):
        self.model = model
        self.total_steps = total_steps
        # 30% of steps for interpolation as per the paper
        self.interpolation_steps = interpolation_steps or int(0.3 * total_steps)
        
    def on_step_begin(self, args, state, control, **kwargs):
        current_step = state.global_step
        
        # Calculate alpha (0 -> 1 over interpolation_steps)
        # This means beta goes from 1 -> 0 (original activation -> bilinear)
        if current_step <= self.interpolation_steps:
            alpha = current_step / self.interpolation_steps
        else:
            alpha = 1.0
            
        # Update all InterpolatedSiLU modules
        count = 0
        # For DDP, we need to access the module attribute to get the actual model
        actual_model = self.model.module if hasattr(self.model, "module") else self.model
        
        for module in actual_model.modules():
            if isinstance(module, InterpolatedSiLU):
                module.alpha = alpha
                count += 1
                
        # Log occasionally (only from rank 0)
        if (current_step % 100 == 0 or current_step == 1) and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.info(f"Step {current_step}/{self.total_steps}: "
                       f"alpha = {alpha:.4f}, beta = {1-alpha:.4f}, "
                       f"updated {count} modules")
            
            # Log to wandb (only from rank 0)
            if wandb.run is not None:
                wandb.log({
                    "alpha": alpha,
                    "beta": 1.0 - alpha,
                    "updated_modules": count
                }, step=current_step)

def setup_distributed(local_rank):
    """Setup distributed training"""
    if local_rank != -1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            return dist.get_rank(), dist.get_world_size()
        else:
            logger.warning("No CUDA available, falling back to CPU")
            dist.init_process_group(backend="gloo")
            return dist.get_rank(), dist.get_world_size()
    return 0, 1  # rank 0, world_size 1 for non-distributed

def main():
    """Main function to run the TinyLlama bilinear finetuning"""
    parser = argparse.ArgumentParser(description="Finetune TinyLlama to a bilinear variant")
    parser.add_argument("--wandb_project", type=str, default="tinyllama-bilinear", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./bilinear_tinyllama_chat", help="Output directory")
    parser.add_argument("--total_tokens", type=int, default=500_000_000, help="Total tokens for training")
    parser.add_argument("--seed", type=int, default=32, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision training")
    # Add distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size = setup_distributed(args.local_rank)
    is_main_process = rank == 0
    
    # Set seeds
    torch.manual_seed(args.seed)
    
    # Initialize wandb (only on main process)
    if is_main_process:
        run_name = f"tinyllama-bilinear-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"Initializing wandb: {args.wandb_project}/{run_name}")
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "model": "TinyLlama_1.1B",
                "total_tokens": args.total_tokens,
                "interpolation_schedule": "linear_30_percent",
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "learning_rate": args.lr,
                "seed": args.seed,
                "mixed_precision": "no" if args.no_mixed_precision else "bf16",
                "num_gpus": world_size
            }
        )
    
    # Load model and tokenizer
    if is_main_process:
        logger.info("Loading TinyLlama model and tokenizer")
    start_time = time.time()
    
    #model_name = "TinyLlama/TinyLlama_v1.1"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model on the current device
    device_map = {"": args.local_rank if args.local_rank != -1 else "cuda" if torch.cuda.is_available() else "cpu"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    
    if is_main_process:
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {total_params:,} parameters")
    
    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if is_main_process:
        logger.info("Loading FineWeb dataset")
    try:
        raw_dataset = load_dataset(
            "HuggingFaceFW/fineweb", 
            split="train", 
            streaming=True
        )
    except Exception as e:
        if is_main_process:
            logger.error(f"Error loading FineWeb dataset: {str(e)}")
            logger.info("Falling back to alternative dataset")
        raw_dataset = load_dataset(
            "manishiitg/fineweb_english", 
            split="train", 
            streaming=True
        )
    
    # Create a debugging or training dataset
    if args.debug:
        # Small batch for debugging
        if is_main_process:
            logger.info("Debug mode: using small sample")
        sample_data = list(raw_dataset.take(10))
        sample_texts = [item['text'] for item in sample_data]
        encodings = tokenizer(
            sample_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        )
        sample_input_ids = encodings.input_ids.to(model.device)
        
        # Run the hook-based replacement in debug mode only
        if is_main_process:
            logger.info("Running hook-based SiLU replacement")
        replacer = HookBasedGateReplacer()
        model = replacer.run_full_replacement(model, sample_input_ids)
        
        # Exit after debugging
        if is_main_process:
            logger.info("Debug completed, exiting")
        return
    
    # Create training dataset with distributed awareness
    train_dataset = TokenStreamingDataset(
        raw_dataset,
        tokenizer,
        max_length=1024,
        target_tokens=args.total_tokens,
        rank=rank,
        world_size=world_size
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Calculate training parameters
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    tokens_per_batch = batch_size * 1024 * grad_accum
    # Total steps is based on tokens per batch across all GPUs
    total_steps = args.total_tokens // (tokens_per_batch * world_size)
    interpolation_steps = int(0.3 * total_steps)
    
    if is_main_process:
        logger.info(f"Training config: {args.total_tokens:,} tokens, {total_steps:,} steps, {world_size} GPUs")
        logger.info(f"First {interpolation_steps:,} steps (30%) will interpolate β to 0")
    
    # Create a sample batch for the replacer
    if is_main_process:
        logger.info("Preparing sample batch for hook-based replacer")
    sample_data = list(raw_dataset.take(10))
    sample_texts = [item['text'] for item in sample_data]
    encodings = tokenizer(
        sample_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024
    )
    sample_input_ids = encodings.input_ids.to(model.device)
    
    # Run the hook-based replacement
    if is_main_process:
        logger.info("Running hook-based SiLU replacement")
    replacer = HookBasedGateReplacer()
    model = replacer.run_full_replacement(model, sample_input_ids)
    
    # Wrap model with DDP
    if args.local_rank != -1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )
    
    # Set up training arguments
    output_dir = args.output_dir
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=total_steps,
        save_steps=min(5000, total_steps // 10),
        save_total_limit=5,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        # Fix: Use bf16 mixed precision for bfloat16 models instead of fp16, or disable mixed precision
        fp16=False,  # Turn off fp16
        bf16=not args.no_mixed_precision,  # Use bf16 if mixed precision is enabled
        learning_rate=args.lr,
        warmup_steps=int(0.01 * total_steps),
        report_to=["wandb"] if is_main_process else [],
        dataloader_num_workers=4,  # Reduced to avoid contention
        remove_unused_columns=False,
        # Distributed training settings
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[
            AlphaDecayCallback(
                model, 
                total_steps=total_steps, 
                interpolation_steps=interpolation_steps
            )
        ]
    )
    
    # Start training
    if is_main_process:
        logger.info("Starting training")
    trainer.train()
    
    # Save final model (only from main process)
    if is_main_process:
        logger.info("Training completed, saving model")
        # Make sure to get the actual model from DDP wrapper
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save config info
        config_info = {
            "model_name": "TinyLlama-1.1B finetuned with bilinear MLP",
            "original_model": model_name,
            "total_tokens": args.total_tokens,
            "interpolation_schedule": "Linearly interpolate β to 0 during first 30% of tokens",
            "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "wandb_run_id": wandb.run.id if wandb.run else None,
            "number_of_gpus": world_size
        }
        
        # Save as text
        with open(f"{output_dir}/bilinear_config.txt", "w") as f:
            for key, value in config_info.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Model saved to {output_dir}")
        logger.info("Finetuning complete!")
        
        # Finish wandb
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Only log from main process to avoid duplicate error messages
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.error(f"Error: {str(e)}", exc_info=True)
            if wandb.run is not None:
                wandb.log({"error": str(e)})
                wandb.finish(exit_code=1)
        # Make sure all processes exit with error
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
