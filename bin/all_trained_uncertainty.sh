#!/bin/bash

# Ensure the script exits on any command failure
set -e

# Define CUDA device
CUDA_DEVICE="2"

for data in s_nq s_trivia s_squad s_hotpotqa s_2wikimultihopqa s_musique; do

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python trained_uncertainty.py \
        --model_path VityaVitalich/Llama3.1-8b-instruct \
        --cache_dir /home/data/v.moskvoretskii/cache/ \
        --data_path data/datasets/${data} \
        --question_column question_text \
        --context_column none \
        --output_column none \
        --batch_size 16 \

done