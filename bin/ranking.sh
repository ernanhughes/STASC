#!/bin/bash
metrics=("LLM Call" "Retrieval Calls" "Inaccuracy" "ROCAUC" "Accuracy" "Correlation")

# Base command options
main_path="data/main.csv"
include_baselines=1
output_dir="logs/ranking/selected_best_eigval"

# Optional models (uncomment if needed)
# selected_models="DRAGIN FLARE AdaptiveRAG Rowen_CL Rowen_CM Rowen_Hybrid Seakr IRCoT FS-RAG 'Max Entropy' EigValLaplacian Lex-Similarity 'Mean Entropy' SAR"

# Create output directory if it doesn't exist
mkdir -p ${output_dir}

# Loop through the metrics and run the command
for metric in "${metrics[@]}"; do
    # Set aggregation based on the metric
    if [[ "$metric" == "LLM Call" || "$metric" == "Retrieval Calls" ]]; then
        aggregation="min"
    else
        aggregation="max"
    fi

    # Replace spaces in metric names with underscores for output filename
    metric_safe=$(echo "$metric" | sed 's/ /_/g')
    
    # Define output file dynamically
    output_file="${output_dir}/${metric_safe}.txt"
    
    # Run the Python script
    echo "Running for target metric: ${metric} with aggregation: ${aggregation}"
    python new_ranking.py \
        --main_path ${main_path} \
        --include_baselines ${include_baselines} \
        --output "${output_file}" \
        --target_metric "${metric}" \
        --aggregation ${aggregation} \
        --selected_models DRAGIN FLARE AdaptiveRAG Rowen_CL Rowen_CM Rowen_Hybrid Seakr IRCoT FS-RAG EigValLaplacian best_uc "All Context" "No Context"
done

# python new_ranking.py \
#     --main_path data/main.csv \
#     --include_baselines 1 \
#     --output 'logs/ranking/overall_LMC.txt' \
#     --target_metric 'LLM Call' \
#     --aggregation 'min' \
#    # --selected_models DRAGIN FLARE AdaptiveRAG Rowen_CL Rowen_CM Rowen_Hybrid Seakr IRCoT FS-RAG "Max Entropy" EigValLaplacian Lex-Similarity "Mean Entropy" SAR
