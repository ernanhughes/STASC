CUDA_VISIBLE_DEVICES=1 python uncertainty.py \
	--model_path VityaVitalich/Llama3-8b-instruct \
    --cache_dir /home/data/v.moskvoretskii/cache/ \
    --data_path data/datasets/nq \
    --question_column question \
    --context_column none \
	--output_column no_context_response \
	--batch_size 4 \
	--save_path data/raw/uc.pt
