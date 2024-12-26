#!/bin/bash

# Ensure the script exits on any command failure
set -e

# Define CUDA device
CUDA_DEVICE="1"

# Variables (modify these to use a different dataset or configuration)
DATASET_NAME="s_musique"
MODEL_PATH="VityaVitalich/Llama3.1-8b-instruct"
CACHE_DIR="/home/data/v.moskvoretskii/cache/"
DATA_PATH="data/datasets/$DATASET_NAME"
RAW_DATA_PATH="data/raw"
CONTEXT_COLUMN="retrieved_contexts"
QUESTION_COLUMN="question_text"
GT_COLUMN="reference"
BATCH_SIZE=32

# Step 1: Generate responses with context
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python generate.py \
  --model_path $MODEL_PATH \
  --output_path $RAW_DATA_PATH/${DATASET_NAME}_context_response \
  --data_path $DATA_PATH \
  --cache_dir $CACHE_DIR \
  --use_context_col $CONTEXT_COLUMN \
  --prompt_type when_to_retrieve \
  --number_output_seq 1 \
  --critic_col none

# Step 2: Generate responses without context
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python generate.py \
  --model_path $MODEL_PATH \
  --output_path $RAW_DATA_PATH/${DATASET_NAME}_no_context_response \
  --data_path $DATA_PATH \
  --cache_dir $CACHE_DIR \
  --use_context_col none \
  --prompt_type when_to_retrieve \
  --number_output_seq 1 \
  --critic_col none

# Step 3: Add generated responses to the dataset for train and test splits
for SPLIT in train test; do
  python add_column.py \
    --dataset_path $DATA_PATH \
    --input_path $RAW_DATA_PATH/${DATASET_NAME}_context_response_${SPLIT}.pickle \
    --new_column_name context_response \
    --keyword assistant

  python add_column.py \
    --dataset_path $DATA_PATH \
    --input_path $RAW_DATA_PATH/${DATASET_NAME}_no_context_response_${SPLIT}.pickle \
    --new_column_name no_context_response \
    --keyword assistant

done

# Step 4: Calculate tokens
python add_token_count.py \
  --dataset_path $DATA_PATH \
  --model_name $MODEL_PATH \
  --context_col $CONTEXT_COLUMN \
  --question_col $QUESTION_COLUMN

# Step 5: Calculate uncertainty
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python uncertainty.py \
  --model_path $MODEL_PATH \
  --cache_dir $CACHE_DIR \
  --data_path $DATA_PATH \
  --question_column $QUESTION_COLUMN \
  --context_column none \
  --output_column none \
  --batch_size $BATCH_SIZE

# Step 6: Analyze uncertainty
python analyze_uncertainty.py \
  --data_path $DATA_PATH \
  --no_context_col no_context_response \
  --with_context_col context_response \
  --gt_col $GT_COLUMN

# Completion message
echo "Pipeline completed successfully for dataset: $DATASET_NAME"
