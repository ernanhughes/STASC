CUDA_VISIBLE_DEVICES=0 python generate.py \
	--model_path VityaVitalich/Llama3.1-8b-instruct \
	--output_path data/raw/s_nq_verb_confidence \
	--data_path data/datasets/s_nq \
	--cache_dir /home/data/v.moskvoretskii/cache/ \
	--use_context_col retrieved_contexts \
	--prompt_type verbalized_confidence_prompt \
	--number_output_seq 1 \
	--critic_col none
