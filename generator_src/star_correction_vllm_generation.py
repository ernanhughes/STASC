import sys
import os


sys.path.append('../')
sys.path.append('./')

os.environ["VLLM_LOGGING_LEVEL"] = 'FATAL'

import argparse
from pathlib import Path
import datasets
from datasets import Dataset, DatasetDict
from functools import partial
from vllm import LLM, SamplingParams
from utils.generation_utils import generate_for_dataset, store_generation_results, load_config, extract_parameters
from prompts.self_correct_star_prompts import star_correction_initial_generation_prompt, star_correction_prompt 
from prompts.prompt_schemas import load_few_shot_prompts
from utils.eval_utils import has_answer
from utils.utils import KM, flatten_predictions
from transformers import AutoTokenizer
import yaml
import subprocess
import torch
import logging

def collect_improving_corrections(
    dataset,
    question_col="question",
    reference_col="reference",
    # Initial Answer column
    inital_answer_col="star_generation_answer",
    # Correction column
    correction_col="star_rationalization_answer",
    id_col="id",
    output_path='temp_dataset',
    strict_improvement=True
):

    system_prompt = (
        "You are a helpful reasoning assistant. "
        "Please reason through the question step by step before giving a final answer."
    )

    instructions = (
        "Generate a short chain-of-thought rationale, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )
    correction_prompt = (
        "Below is the question and the initial answer. "
        "Generate a correction to the initial answer if it is incorrect"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # Prepare lists for final flattened data
    new_ids = []
    new_questions = []
    new_refs = []
    new_corrections = []
    new_answers = []
    new_messages = []

    # Iterate over each row
    for row in dataset:
        row_id = row[id_col]
        question = row[question_col]
        reference = row[reference_col]

        # 1) Retrieve generation answers/rationales
        for init_answer in row[inital_answer_col]:
            for correction in flatten_predictions(row[correction_col]):

                # 3) Check if there is an improvement
                init_is_correct = has_answer(reference, init_answer)
                correction_is_correct = has_answer(reference, correction)

                if strict_improvement:
                    use_sample = correction_is_correct > init_is_correct
                else:
                    use_sample = init_is_correct and correction_is_correct

                if use_sample:
                    new_ids.append(f"{row_id}_gen")
                    new_questions.append(question)
                    new_refs.append(reference)
                    new_answers.append([init_answer])
                    new_corrections.append([correction])

                    user_question = f"Question:\n{question}\n\nReason step by step, then conclude with the answer."

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": instructions},
                        {"role": "user", "content": user_question},
                        {"role": "user", "content": init_answer},
                        {"role": "user", "content": correction_prompt},
                        {"role": "assistant", "content": correction}

                    ]
                    new_messages.append(messages)


    # Build the new dictionary
    flattened_data = {
        id_col: new_ids,
        question_col: new_questions,
        reference_col: new_refs,
        inital_answer_col: new_answers,
        correction_col: new_corrections,
        "messages": new_messages,
    }

    print(f'[INFO] Filtered {len(new_ids)} Corrections')


    # Convert to a new HF Dataset
    flattened_dataset = DatasetDict({"train": Dataset.from_dict(flattened_data)})
    flattened_dataset.save_to_disk(output_path)

    print(f'[INFO] Saved filtered corrections to {output_path}')

    return flattened_dataset


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--generation_model_path", type=str, required=True)
    parser.add_argument("--ft_dataset_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--initial_generation", action="store_true", help="Set this flag for initial generation")
    args = parser.parse_args()

    iteration = args.iteration

    config = load_config(args.config_path)

    # Load dataset
    dataset = datasets.load_from_disk(str(config['data_path']))
    train_data, test_data = dataset["train"], dataset["test"]


    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], cache_dir=config['cache_dir'])


    star_correction_initial_generation_few_shot = load_few_shot_prompts(config['few_shot_dir'], 'star_correction_initial_generation')
    star_correction_few_shot = load_few_shot_prompts(config['few_shot_dir'], 'star_correction')
    


    # Prompt functions
    initial_generation_prompt_func = partial(
        star_correction_initial_generation_prompt,
        tokenizer=tokenizer,
        question_col=config['question_col'],
        few_shot_prompts=star_correction_initial_generation_few_shot,
    )

    correction_prompt_func = partial(
        star_correction_prompt,
        tokenizer=tokenizer,
        question_col=config['question_col'],
        few_shot_prompts=star_correction_few_shot,
        initial_answer_col='star_correction_initial_generation'
    )



    # Initialize model (M0)
    model = LLM(
        args.generation_model_path,
        download_dir=config['cache_dir'],
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=config['gpu_memory_utilization'],
        enforce_eager=config['enforce_eager'],
        max_model_len=config['max_model_len'],
        disable_log_stats=True,  # Disables logging statistics
        seed=config['random_seed']
        #disable_log_requests=True,  # Disables logging requests
    )
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=config['temperature'],
        top_p=config['top_p'],
        max_tokens=config['max_tokens'],
        n=1,
        seed=config['random_seed'],
    )

    # if generating with initial model and now generating initial answers
    if (not config['initial_answer_with_new_model']) and (args.initial_generation):

        print(f"[INFO] Starting generation of initial answers at {iteration}/{config['num_star_iterations']}")
        sampling_params.n = config['number_output_initial_generations']
        train_data = perform_generation(
            data=train_data,
            model=model,
            prompt_func=initial_generation_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col='star_correction_initial_generation'
        )

        sampling_params.n = 1
        test_data = perform_generation(
            data=test_data,
            model=model,
            prompt_func=initial_generation_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col='star_correction_initial_generation'
        )

        train_acc = KM(train_data, target_col=f'star_correction_initial_generation', gt_col=config['gold_col'])
        test_acc = KM(test_data, target_col=f'star_correction_initial_generation', gt_col=config['gold_col'])

        print(f'[INFO] Initial Train Accuracy {train_acc}')
        print(f'[INFO] Initial Test Accuracy {test_acc}')

        dataset = DatasetDict({"train": train_data, "test": test_data})
        dataset.save_to_disk(args.ft_dataset_path)

        print(f'[INFO] Saving initial generations to {args.ft_dataset_path}')

        return
    
    # if generating with initial model and now generating corrections
    if (not config['initial_answer_with_new_model']) and (not args.initial_generation):

        print(f"[INFO] Starting generation of corrections at {iteration}/{config['num_star_iterations']}")

        sampling_params.n = config['number_output_corrections']
        train_data = perform_generation(
            data=train_data,
            model=model,
            prompt_func=correction_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col=f'star_correction_{iteration}'
        )

        sampling_params.n = 1
        test_data = perform_generation(
            data=test_data,
            model=model,
            prompt_func=correction_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col=f'star_correction_{iteration}'
        )

        train_acc = KM(train_data, target_col=f'star_correction_{iteration}', gt_col=config['gold_col'])
        test_acc = KM(test_data, target_col=f'star_correction_{iteration}', gt_col=config['gold_col'])

        print(f'[INFO] Correction at step {iteration} Train Accuracy {train_acc}')
        print(f'[INFO] Correction at step {iteration} Test Accuracy {test_acc}')

        collect_improving_corrections(
            dataset=train_data,
            question_col=config['question_col'],
            reference_col=config['gold_col'],
            inital_answer_col='star_correction_initial_generation',
            correction_col=f'star_correction_{iteration}',
            id_col=config['id_col'],
            output_path=args.ft_dataset_path,
            strict_improvement=config['only_better_correction']
        )
        test_data.save_to_disk(f"{args.ft_dataset_path}_test")
        return
    

    # if generating with latest model both initial answers and corrections

    print(f"[INFO] Starting generation of initial answers at {iteration}/{config['num_star_iterations']}")

    sampling_params.n = config['number_output_initial_generations']
    train_data = perform_generation(
        data=train_data,
        model=model,
        prompt_func=initial_generation_prompt_func,
        sampling_params=sampling_params,
        id_key=config['id_col'],
        output_col='star_correction_initial_generation'
    )

    sampling_params.n = 1
    test_data = perform_generation(
        data=test_data,
        model=model,
        prompt_func=initial_generation_prompt_func,
        sampling_params=sampling_params,
        id_key=config['id_col'],
        output_col='star_correction_initial_generation'
    )
    train_acc = KM(train_data, target_col=f'star_correction_initial_generation', gt_col=config['gold_col'])
    test_acc = KM(test_data, target_col=f'star_correction_initial_generation', gt_col=config['gold_col'])

    print(f'[INFO] Initial Train Accuracy {train_acc}')
    print(f'[INFO] Initial Test Accuracy {test_acc}')

    print(f"[INFO] Starting generation of corrections at {iteration}/{config['num_star_iterations']}")

    sampling_params.n = config['number_output_corrections']
    train_data = perform_generation(
            data=train_data,
            model=model,
            prompt_func=correction_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col=f'star_correction_{iteration}'
    )

    sampling_params.n = 1
    test_data = perform_generation(
            data=test_data,
            model=model,
            prompt_func=correction_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col=f'star_correction_{iteration}'
    )

    train_acc = KM(train_data, target_col=f'star_correction_{iteration}', gt_col=config['gold_col'])
    test_acc = KM(test_data, target_col=f'star_correction_{iteration}', gt_col=config['gold_col'])

    print(f'[INFO] Correction at step {iteration} Train Accuracy {train_acc}')
    print(f'[INFO] Correction at step {iteration} Test Accuracy {test_acc}')

    collect_improving_corrections(
        dataset=train_data,
        question_col=config['question_col'],
        reference_col=config['gold_col'],
        inital_answer_col='star_correction_initial_generation',
        correction_col=f'star_correction_{iteration}',
        id_col=config['id_col'],
        output_path=args.ft_dataset_path,
        strict_improvement=config['only_better_correction']
    )
    
    test_data.save_to_disk(f"{args.ft_dataset_path}_test")

if __name__ == '__main__':
    main()
