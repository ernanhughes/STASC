
# Model Configuration
model_path: "luezzka/Llama-3.2-1B-Instruct"
cache_dir: "/home/data/v.moskvoretskii/cache/"
gpu_memory_utilization: 0.9       # GPU memory utilization (0.0 to 1.0)
enforce_eager: True                # Whether to enforce eager execution
max_model_len: 12196                # Maximum model length

# Dataset Configuration
data_path: "data/datasets/s_trivia"
id_col: "question_id"                        # Unique identifier column in the dataset
question_col: "question_text"            # Column containing the question text
gold_col: "reference"

num_refine_iterations: 5
run_name: "trivia_seed_1_llama-1b_5"

# Generation Configuration
few_shot_dir: "few_shots"            # Directory containing few-shot JSON files
number_output_seq: 1                 # Number of sequences to generate per prompt
random_seed: 1

# Sampling Parameters
temperature: 0.6                     # Sampling temperature
top_p: 0.9                           # Top-p (nucleus) sampling
max_tokens: 1024                      # Maximum number of tokens to generate
