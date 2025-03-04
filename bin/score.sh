export ACCELERATE_CONFIG='configs/accelerate_config.yaml'
export SCORE_CONFIG='configs/score.yaml'
export WANDB_API_KEY=''


accelerate launch --config_file $ACCELERATE_CONFIG score.py --config_path $SCORE_CONFIG