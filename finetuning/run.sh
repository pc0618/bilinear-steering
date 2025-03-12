#!/bin/bash

# Set default values
MODEL="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATASET="togethercomputer/RedPajama-Data-1T-Sample"
OUTPUT_DIR="./results"
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=5e-5
MAX_STEPS=10000
BETA_DECAY=0.3
DECAY_FN="linear"
USE_WANDB=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad_accum)
      GRAD_ACCUM="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --beta_decay)
      BETA_DECAY="$2"
      shift 2
      ;;
    --decay_fn)
      DECAY_FN="$2"
      shift 2
      ;;
    --no_wandb)
      USE_WANDB=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Build the command
CMD="python tinyllama_finetune.py \
  --model_name_or_path=$MODEL \
  --dataset_name=$DATASET \
  --output_dir=$OUTPUT_DIR \
  --per_device_train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACCUM \
  --learning_rate=$LEARNING_RATE \
  --max_steps=$MAX_STEPS \
  --beta_decay_point=$BETA_DECAY \
  --beta_decay_function=$DECAY_FN \
  --fp16 \
  --logging_steps=10 \
  --save_steps=1000 \
  --save_total_limit=2"

# Add wandb if enabled
if [ "$USE_WANDB" = true ]; then
  CMD="$CMD --report_to=wandb"
fi

# Print the command
echo "Running: $CMD"

# Execute the command
eval $CMD