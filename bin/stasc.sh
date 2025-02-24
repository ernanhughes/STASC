export WANDB_API_KEY='<key>'

CUDA_VISIBLE_DEVICES=0,2 python star_correction.py \
    --config configs/self_correction_star_config.yaml \
    --ft_config configs/fine_tune.yaml \
    --accelerate_config_path configs/accelerate_config.yaml 
