import argparse
from pathlib import Path
from utils.generation_utils import load_config
import os
import yaml
import subprocess
import threading
import logging


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

class InfoOnlyFilter(logging.Filter):
    """A filter to allow only messages containing '[INFO]' in them."""
    def filter(self, record):
        return "[INFO]" in record.getMessage()

def setup_logger(run_name: str, log_all="logs/all_logs.log", log_info="logs/info_logs.log"):
    """
    Sets up a logger that writes:
    - All logs to `log_all`
    - Only logs explicitly containing `[INFO]` to `log_info`
    - Logs to console
    """
    logger_name = f"self_correct_STaR_logger_{run_name}"  
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if already set
    if not logger.handlers:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))

        # File handler for ALL logs
        file_handler_all = logging.FileHandler(log_all, mode="a", encoding="utf-8")
        file_handler_all.setLevel(logging.INFO)
        file_handler_all.setFormatter(logging.Formatter(log_format))

        # File handler for INFO logs with "[INFO]"
        file_handler_info = logging.FileHandler(log_info, mode="a", encoding="utf-8")
        file_handler_info.setLevel(logging.INFO)
        file_handler_info.setFormatter(logging.Formatter(log_format))
        file_handler_info.addFilter(InfoOnlyFilter())  # Only log messages containing "[INFO]"

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler_all)
        logger.addHandler(file_handler_info)

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


    # Ensure all log directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/detailed", exist_ok=True)
    os.makedirs("logs/general", exist_ok=True)

    logger = setup_logger(
        config['run_name'], 
        log_all=f"logs/detailed/Self_Correct_STaR_{config['run_name']}.log",
        log_info=f"logs/general/Self_Correct_STaR_{config['run_name']}.log"
    )

    # Define the base directory for all runs
    stasc_dir = os.path.join(config['cache_dir'], 'STaSC')
    os.makedirs(stasc_dir, exist_ok=True)  # Ensure the main STaSC directory exists

    # Define the directory for the specific run
    run_dir = os.path.join(stasc_dir, config['run_name'])
    os.makedirs(run_dir, exist_ok=True)  # Ensure the run directory exists

    # Create iteration directories (iter_0, iter_1, ...)
    iteration_dirs = [os.path.join(run_dir, f"iter_{i}") for i in range(config['num_star_iterations'] + 1)]
    for iter_dir in iteration_dirs:
        os.makedirs(iter_dir, exist_ok=True)  # Ensure iteration directories exist



    # start from initial model
    generation_model_path = config['model_path']
    ft_config['model']['model_name_or_path'] = generation_model_path


    # Generate initial answers with initial model if need so
    if not config['initial_answer_with_new_model']:
        
        initial_ans_dataset_path = os.path.join(run_dir, f"initial_data")  # Get current iteration dir
        updated_config_path = f"configs/temp_self_corr_STaR_config_{config['run_name']}.yaml"
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
        logger.info(f"[INFO] Starting iteration {iteration}/{config['num_star_iterations']}")

        # Create name for new dataset
      #  ft_dataset_path = f"{data_dir}/data_{iteration}"

        iter_dir = os.path.join(run_dir, f"iter_{iteration}")  # Get current iteration dir
        
        # Define model and data paths for the iteration
        ft_dataset_path = os.path.join(iter_dir, 'data')

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
        generation_model_path = os.path.join(iter_dir, 'model')
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

        if not config['train_from_initial_model']:
            ft_config['model']['model_name_or_path'] = generation_model_path

        # End of iteration; M_{n} is now your updated model

    logger.info("[INFO] Self-Correction STaR algorithm completed.")

if __name__ == "__main__":
    main()
