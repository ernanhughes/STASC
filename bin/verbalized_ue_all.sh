#!/bin/bash

# List of datasets
datasets=("s_nq" "s_trivia" "s_squad" "s_hotpotqa" "s_2wikimultihopqa" "s_musique")

# Loop through datasets and run the same script with each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"

    # Generate the specific config file for the current dataset
    config_file="config_${dataset}.yaml"

    # Create the config file with dynamic dataset path
    cat <<EOL > "$config_file"
# Model Configuration
model_path: "VityaVitalich/Llama3.1-8b-instruct"
cache_dir: "/home/data/v.moskvoretskii/cache/"
gpu_memory_utilization: 0.8        # GPU memory utilization (0.0 to 1.0)
enforce_eager: True                # Whether to enforce eager execution
max_model_len: 12288               # Maximum model length

# Dataset Configuration
data_path: "data/datasets/$dataset"  # Dataset path specific to current dataset
id_col: "question_id"                # Unique identifier column in the dataset
question_col: "question_text"        # Column containing the question text
use_context_col: "none"              # Column name for context or "none" if not used
answer_col: "no_context_response"    # Column containing existing answers
critic_col: "no_context_response_critic_2shot"  # Column with criticisms

# Generation Configuration
prompt_type: "verbalized_1s_topk"  # Options: generate, critic, revise, etc.
few_shot_dir: "few_shots"             # Directory containing few-shot JSON files
result_col: "verbalized_1s_top2_answer"   # New column name to store generation results
number_output_seq: 1                 # Number of sequences to generate per prompt
verbalized_top_k: 1                             # Verbalized Top-K 1S

# Sampling Parameters
temperature: 0.6                     # Sampling temperature
top_p: 0.9                           # Top-p (nucleus) sampling
max_tokens: 256                      # Maximum number of tokens to generate
EOL

    # Run the model with the generated config file for the current dataset
    python generate.py --config "$config_file"

    echo "Finished processing dataset: $dataset"
    
    # Optional: Remove the generated config file after use
    rm "$config_file"
done