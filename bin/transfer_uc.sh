python transfer_uc_performance.py \
    --data_paths data/datasets/s_nq data/datasets/s_trivia data/datasets/s_squad data/datasets/s_hotpotqa data/datasets/s_2wikimultihopqa data/datasets/s_musique \
    --no_context_col new_retriever_adaptive_rag_no_retrieve \
    --with_context_col new_retriever_adaptive_rag_one_retrieve \
    --gt_col reference