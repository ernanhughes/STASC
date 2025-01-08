import argparse
from pathlib import Path
import datasets
from datasets import Dataset, DatasetDict
from functools import partial
from generation_utils import generate_for_dataset, store_generation_results, load_config, extract_parameters
from self_refine_prompts import gather_full_conversation, initial_generation_prompt, refinement_prompt, feedback_prompt
from eval_utils import has_answer
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer
from prompt_schemas import load_few_shot_prompts
import os
import yaml
import subprocess
import threading
import torch
import logging

# TODO:
# break when found "it is correct", add this to the feedback
# add evaluation
# track length
# Add CoT Maybe
# Add explicit Feedback
# add self-consistency



def setup_logger(run_name: str, log_file="star.log"):
    """
    Sets up a logger named "star_logger_{run_name}" that writes both 
    to the console and to `log_file`.
    """
    logger_name = f"star_logger_{run_name}"    # e.g. "star_logger_test_0"
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

def perform_generation(data, model, prompt_func, sampling_params, id_key, output_col):
    """
    Perform (rationale) generation or (rationalization) generation for the dataset.
    Store the generation results in the dataset under 'output_col'.
    """
    generation_results = generate_for_dataset(
        model=model,
        data=data,
        prompt_function=prompt_func,
        sampling_params=sampling_params,
        id_key=id_key
    )
    return store_generation_results(data, generation_results, result_col=output_col, id_col=id_key)



def main():
    parser = argparse.ArgumentParser(description="Run the Self-Refine Algorithm")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)
    logger = setup_logger(config['run_name'], log_file=f"self_refine_{config['run_name']}.log")

    # Load dataset
    dataset = datasets.load_from_disk(str(config['data_path']))
    train_data, test_data = dataset["train"], dataset["test"]


    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], cache_dir=config['cache_dir'])

    # few shots
    generation_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'self_refine_generation')
    feedback_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'self_refine_feedback')
    refine_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'self_refine_refinement')


    # Prompt functions
    conversation_gather_func = partial(
        gather_full_conversation, 
        question_col=config['question_col'],
        generation_col='self_refine_initial_generation',
        feedback_prefix='self_refine_feedback_',
        refinement_prefix='self_refine_refinement_')

    initial_generation_prompt_func = partial(
        initial_generation_prompt,
        tokenizer=tokenizer,
        question_col=config['question_col'],
        few_shot_prompts=generation_few_shot_prompts,
    )

    feedback_prompt_func = partial(
        feedback_prompt,
        tokenizer=tokenizer,
        conversation_gather_func=conversation_gather_func,
        few_shot_prompts=feedback_few_shot_prompts,
    )
    refine_prompt_func = partial(
        refinement_prompt,
        tokenizer=tokenizer,
        conversation_gather_func=conversation_gather_func,
        few_shot_prompts=refine_few_shot_prompts,
    )

    # Initialize model (M0)
    model = LLM(
        config['model_path'],
        download_dir=config['cache_dir'],
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=config['gpu_memory_utilization'],
        enforce_eager=config['enforce_eager'],
        max_model_len=config['max_model_len'],
        # disable_log_stats=True,  # Disables logging statistics
        #disable_log_requests=True,  # Disables logging requests
    )
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=config['temperature'],
        top_p=config['top_p'],
        max_tokens=config['max_tokens'],
        n=config['number_output_seq']
    )

    train_data = perform_generation(
        data=train_data,
        model=model,
        prompt_func=initial_generation_prompt_func,
        sampling_params=sampling_params,
        id_key=config['id_col'],
        output_col=f"self_refine_initial_generation"
    )


    # Outer loop: for n in 1...N
    for iteration in range(config['num_refine_iterations']):
        logger.info(f"Starting feedback {iteration + 1}/{config['num_refine_iterations']}")

        train_data = perform_generation(
            data=train_data,
            model=model,
            prompt_func=feedback_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col=f"self_refine_feedback_{iteration}"  # store model's answer after rationale generation
        )

        logger.info(f"Starting refinement {iteration + 1}/{config['num_refine_iterations']}")


        train_data = perform_generation(
            data=train_data,
            model=model,
            prompt_func=refine_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col=f"self_refine_refinement_{iteration}"
        )


        # End of iteration; M_{n} is now your updated model

    logger.info("Self-Refine algorithm completed.")

if __name__ == "__main__":
    main()
