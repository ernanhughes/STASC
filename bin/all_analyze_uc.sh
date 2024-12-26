#!/bin/bash

# Ensure the script exits on any command failure
set -e


for data in s_nq s_trivia s_squad s_hotpotqa s_2wikimultihopqa s_musique; do

    python analyze_uncertainty.py \
    --data_path data/datasets/${data} \
    --no_context_col new_retriever_adaptive_rag_no_retrieve \
    --with_context_col new_retriever_adaptive_rag_one_retrieve \
    --gt_col reference \
    --out_name adarag_uc_${data}

done