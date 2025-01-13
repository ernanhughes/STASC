import argparse
from pathlib import Path
from utils.generation_utils import load_config
import os
import yaml
import subprocess
import threading
import logging


# TODO:
# debug
# maybe need to truncate correct rationales number
# add files with few shot
# add evaluation
# write to logger



def _stream_subprocess_output(pipe, log_func):
    """
    Reads lines from a subprocess pipe (stdout or stderr) and calls log_func(line) in real time.
    """
    for line in iter(pipe.readline, ""):
        if line:
            log_func(line.rstrip("\n"))
    pipe.close()

def run_subprocess_in_real_time(cmd, logger):
    """
    Starts a subprocess with real-time capture of stdout and stderr.
    Logs each line immediately via the provided logger.
    
    Returns:
      The subprocess's return code (0 if success, non-zero if an error occurred).
    """
    logger.info("Running command: %s", " ".join(cmd))
    
    # Create the Popen object with pipes for stdout/stderr
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Threads to consume stdout/stderr in real time
    stdout_thread = threading.Thread(target=_stream_subprocess_output, args=(process.stdout, logger.info))
    stderr_thread = threading.Thread(target=_stream_subprocess_output, args=(process.stderr, logger.error))

    # Start the threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to finish
    process.wait()
    # Wait for threads to finish reading
    stdout_thread.join()
    stderr_thread.join()

    return process.returncode

def setup_logger(run_name: str, log_file="star.log"):
    """
    Sets up a logger named "star_logger_{run_name}" that writes both 
    to the console and to `log_file`.
    """
    logger_name = f"self_correct_STaR_logger_{run_name}"    # e.g. "star_logger_test_0"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if already set
    if not logger.handlers:
        # 1) Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # 2) File handler
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger



def call_fine_tune(config_yaml_path: str, accelerate_config_path: str, logger: logging.Logger):
    """
    Calls fine_tune.py with the specified YAML config, logging output in real time.
    """
    cmd = [
        "accelerate", "launch",
        "--config_file", accelerate_config_path,
        "fine_tune.py",
        "--config_path", config_yaml_path
    ]
    returncode = run_subprocess_in_real_time(cmd, logger)

    if returncode != 0:
        raise RuntimeError(f"fine_tune.py failed with exit code {returncode}")

def call_vllm_generation(config_path, generation_model_path, ft_dataset_path, iteration, initial_generation, logger: logging.Logger):
    """
    Spawns a new Python process (star_vllm_generation.py) in real time.
    """
    cmd = [
        "python", "generator_src/star_correction_vllm_generation.py",
        "--config_path", config_path,
        "--generation_model_path", generation_model_path,
        "--ft_dataset_path", ft_dataset_path,
        "--iteration", iteration,
    ]

    if initial_generation:
        cmd.append("--initial_generation")
    returncode = run_subprocess_in_real_time(cmd, logger)

    if returncode != 0:
        raise RuntimeError(f"star_vllm_generation.py failed with exit code {returncode}")





def main():
    parser = argparse.ArgumentParser(description="Run the STaR Algorithm")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--ft_config", type=str, required=True, help="Path to the Fine-Tuning YAML config file.")
    parser.add_argument("--accelerate_config_path", type=str, required=True, help="Path to the accelerate YAML config file.")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)
    ft_config = load_config(args.ft_config)

    logger = setup_logger(config['run_name'], log_file=f"logs/Self_Correct_STaR_{config['run_name']}.log")

    # create a path to save models
    star_dir = os.path.join(config['cache_dir'], 'Self_Correct_STaR')
    model_checkpoint_dir = os.path.join(star_dir, config['run_name'])
    data_dir = os.path.join(model_checkpoint_dir, 'data')

    # Recursively create all directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)


    # start from initial model
    generation_model_path = config['model_path']
    ft_config['model']['model_name_or_path'] = generation_model_path


    # Generate initial answers with initial model if need so
    if not config['initial_answer_with_new_model']:
        
        initial_ans_dataset_path = f"{data_dir}/data_initial"
        updated_config_path = "configs/temp_self_corr_STaR_config.yaml"
        call_vllm_generation(
            config_path=args.config,
            generation_model_path=generation_model_path,
            ft_dataset_path=initial_ans_dataset_path,
            iteration=str(0),
            initial_generation=True,
            logger=logger
        )
        config['data_path'] = initial_ans_dataset_path
        with open(updated_config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
    else:
        updated_config_path = args.config


    # Outer loop: for n in 1...N
    for iteration in range(config['num_star_iterations']+1):
        logger.info(f"[INFO] Starting iteration {iteration + 1}/{config['num_star_iterations']}")

        # Create name for new dataset
        ft_dataset_path = f"{data_dir}/data_{iteration}"

        # Call vllm
        call_vllm_generation(
                    config_path=updated_config_path,
                    generation_model_path=generation_model_path,
                    ft_dataset_path=ft_dataset_path,
                    iteration=str(iteration),
                    initial_generation=False,
                    logger=logger
                )

        # to evaluate the last iteration after the fine-tuning
        if iteration == config['num_star_iterations']:
            break


        # name for future model and modified configs
        generation_model_path = f"{model_checkpoint_dir}_{iteration}"
        ft_config["data"]["dataset_name"] = ft_dataset_path
        ft_config["training"]["output_dir"] = generation_model_path
        ft_config['training']['run_name'] = f"{config['run_name']}_{iteration}"

        updated_ft_config_path = "configs/temp_self_corr_STaR_ft_config.yaml"
        with open(updated_ft_config_path, "w") as f:
            yaml.dump(ft_config, f, sort_keys=False)

        # (7) Fine-tune on the combined correct solutions
        call_fine_tune(
            config_yaml_path=updated_ft_config_path,
            accelerate_config_path=args.accelerate_config_path,
            logger=logger
        )

        # End of iteration; M_{n} is now your updated model

    logger.info("[INFO] Self-Correction STaR algorithm completed.")

if __name__ == "__main__":
    main()
