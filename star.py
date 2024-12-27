import argparse
from pathlib import Path
import datasets
from datasets import Dataset
from functools import partial
from vllm import LLM, SamplingParams
from generation_utils import generate_for_dataset, store_generation_results, load_config, extract_parameters
from star_prompts import star_rationale_generation_prompt, star_rationalization_prompt
from eval_utils import has_answer
from transformers import AutoTokenizer
from prompt_schemas import load_few_shot_prompts
import os
import yaml
import subprocess

# TODO:
# add proper tuning pipeline
# prepare proper label and question for fine-tuning
# setup config
# add files with few shot


def call_fine_tune(
    config_yaml_path: str,
    accelerate_config_path: str
):
    """
    Calls fine_tune.py with the specified YAML config.

    Args:
        config_yaml_path (str): Path to your config.yaml file.
        accelerate_config_path (str, optional): Path to accelerate_config.yaml.
            If None, uses the default accelerate config or single-GPU mode.
    """

    # Basic command: "accelerate launch fine_tune.py --config_path config_yaml_path"
    cmd = ["accelerate", "launch"]

    cmd += ["--config_file", accelerate_config_path]

    # Add script + arguments
    cmd += ["fine_tune.py", "--config_path", config_yaml_path]

    print("Running command:", " ".join(cmd))

    # Actually run the command
    subprocess.run(cmd, check=True)

def split_rationale_and_final_answer(generated_text: str):
    """
    Splits a STaR-style generation into two parts:
      1) The rationale (everything after 'Step-by-step reasoning:' until 'Final Answer:')
      2) The final answer (everything after 'Final Answer:')
    If either marker is missing, we do a best-effort parse.
    """
    rationale_marker = "Step-by-step reasoning:"
    answer_marker = "Final Answer:"

    rationale = ""
    final_ans = ""

    text = generated_text.replace("\r", "")  # normalize newlines if desired

    # Locate the markers
    rationale_start_idx = text.find(rationale_marker)
    answer_start_idx = text.find(answer_marker)

    if rationale_start_idx != -1:
        rationale_start = rationale_start_idx + len(rationale_marker)
        if answer_start_idx != -1 and answer_start_idx > rationale_start:
            rationale = text[rationale_start:answer_start_idx].strip()
        else:
            # If there's no final_answer marker or it's out of order
            rationale = text[rationale_start:].strip()

    if answer_start_idx != -1:
        answer_start = answer_start_idx + len(answer_marker)
        final_ans = text[answer_start:].strip()

    return rationale, final_ans

def add_rationale_split(dataset, answers_col, rationale_col_name, final_ans_col_name):

    def split_rationales(example):
        answers = example[answers_col]

        rationales = []
        final_ans_ls = []
        for ans in answers:
            rationale, final_ans = split_rationale_and_final_answer(ans)
            rationales.append(rationale)
            final_ans_ls.append(final_ans)

        example[rationale_col_name] = rationales
        example[final_ans_col_name] = final_ans_ls

        return example

    return dataset.map(split_rationales)

def flatten_correct_rationales(
    dataset,
    question_col="question",
    reference_col="reference",
    # Generation columns
    gen_answer_col="star_generation_answer",
    gen_rationale_col="star_generation_rationale",
    # Rationalization columns
    rat_answer_col="star_rationalization_answer",
    rat_rationale_col="star_rationalization_rationale",
    id_col="id"
):
    """
    For each row in `dataset`, we do:

    1) Check if there's at least one correct plain generation answer in `gen_answer_col`.
       - If yes, collect all (answer, rationale) pairs from generation that are correct.
         (We ignore rationalization.)
    2) Otherwise, collect all (answer, rationale) pairs from rationalization that are correct.

    We create a new row for EACH correct pair:
      {
        id, question, reference, rationale, answer
      }

    Returns a brand-new HF Dataset with these flattened rows.
    """

    # Prepare lists for final flattened data
    new_ids = []
    new_questions = []
    new_refs = []
    new_rationales = []

    # Iterate over each row
    for row in dataset:
        row_id = row[id_col]
        question = row[question_col]
        reference = row[reference_col]

        # 1) Retrieve generation answers/rationales
        gen_answers = row[gen_answer_col]
        gen_rationales = row[gen_rationale_col]

        # 2) Retrieve rationalization answers/rationales
        rat_answers = row[rat_answer_col]
        rat_rationales = row[rat_rationale_col]

        # 3) Check if generation has at least one correct
        gen_correct_indices = [
            i for i, ans in enumerate(gen_answers)
            if has_answer(reference, ans)
        ]
        use_generation = (len(gen_correct_indices) > 0)

        if use_generation:
            # We only take the correct generation pairs
            # (5) Filter correct rationales from the rationale generation step (where pred_answer == gold label)
            # Dn = { (x_i, rhat_i, y_i) | yhat_i == y_i }
            for i in gen_correct_indices:
                new_ids.append(f"{row_id}_gen_{i}")
                new_questions.append(question)
                new_refs.append(reference)
                new_rationales.append(gen_rationales[i])
        else:
            # 4) If none correct in generation, we look at rationalization
            
            # (6) Filter correct rationales from the rationalization step
            # Drat_n = { (x_i, rrat_i, y_i) | yhat_i != y_i AND yhat_rat_i == y_i }
            # So we keep only those that were incorrect before but now correct
            rat_correct_indices = [
                i for i, ans in enumerate(rat_answers)
                if has_answer(reference, ans)
            ]
            for i in rat_correct_indices:
                new_ids.append(f"{row_id}_rat_{i}")
                new_questions.append(question)
                new_refs.append(reference)
                new_rationales.append(rat_rationales[i])

    # Build the new dictionary
    flattened_data = {
        id_col: new_ids,
        question_col: new_questions,
        reference_col: new_refs,
        "rationale": new_rationales,
    }

    # Convert to a new HF Dataset
    flattened_dataset = Dataset.from_dict(flattened_data)
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
    parser = argparse.ArgumentParser(description="Run the STaR Algorithm")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--ft_config", type=str, required=True, help="Path to the Fine-Tuning YAML config file.")
    parser.add_argument("--accelerate_config_path", type=str, required=True, help="Path to the accelerate YAML config file.")

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_dict = load_config(config_path)
    config = extract_parameters(config_dict)

    ft_config = load_config(args.ft_config)

    # create a path to save models
    os.mkdir( f"{config.cache_dir}STaR") 
    model_checkpoint_dir = f"{config.cache_dir}STaR/{config.run_name}"


    # start from initial model
    generation_model_path = config.model_path

    # Load dataset
    dataset = datasets.load_from_disk(str(config.data_path))
    train_data, test_data = dataset["train"], dataset["test"]


    tokenizer = AutoTokenizer.from_pretrained(config.model_path, cache_dir=config.cache_dir)

    # few shots

    generation_few_shot_prompts = load_few_shot_prompts(config.few_shot_dir, 'star_generation')
    rationalization_few_shot_prompts = load_few_shot_prompts(config.few_shot_dir, 'star_rationalization')

    # Prompt functions
    # 1) Rationale generation prompt (Step 3)
    rationale_prompt_func = partial(
        star_rationale_generation_prompt,
        tokenizer=tokenizer,
        question_col=config.question_col,
        few_shot_prompts=generation_few_shot_prompts,
    )
    # 2) Rationalization prompt that includes the gold answer as a hint (Step 4)
    rationalization_prompt_func = partial(
        star_rationalization_prompt,
        tokenizer=tokenizer,
        question_col=config.question_col,
        gold_col=config.gold_col,
        few_shot_prompts=rationalization_few_shot_prompts,
    )

    # Outer loop: for n in 1...N
    for iteration in range(config.num_star_iterations):

        # Initialize model (M0)
        model = LLM(
            model_path=generation_model_path,
            cache_dir=config.cache_dir,
            gpu_memory_utilization=config.gpu_memory_utilization
        )
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            n=config.number_output_seq
        )
        print(f"[INFO] Starting iteration {iteration + 1}/{config.num_iterations}")

        # (3) Generate rationales for all samples: (rhat_i, yhat_i) = M_{n-1}(x_i)
        # We'll store the predicted final answer in 'result_col' or a new column like "pred_answer"
        train_data = perform_generation(
            data=train_data,
            model=model,
            prompt_func=rationale_prompt_func,
            sampling_params=sampling_params,
            id_key=config.id_col,
            output_col=f"star_generation_{iteration}"  # store model's answer after rationale generation
        )
        train_data = add_rationale_split(
            train_data, 
            answers_col=f"star_generation_{iteration}", 
            rationale_col_name=f"star_generation_rationale_{iteration}",
            final_ans_col_name=f"star_generation_answer_{iteration}"
        )


        # (4) Rationalization for all samples, but the prompt includes the gold answer
        # We'll store this rationalized final answer in a new column, e.g., "rationalized_answer"
        train_data = perform_generation(
            data=train_data,
            model=model,
            prompt_func=rationalization_prompt_func,
            sampling_params=sampling_params,
            id_key=config.id_col,
            output_col=f"star_rationalization_{iteration}"
        )
        train_data = add_rationale_split(
            train_data, 
            answers_col=f"star_rationalization_{iteration}", 
            rationale_col_name=f"star_rationalization_rationale_{iteration}",
            final_ans_col_name=f"star_rationalization_answer_{iteration}"
        )

        fine_tuning_data_path = flatten_correct_rationales(
            dataset=train_data,
            question_col=config.question_col,
            reference_col=config.gold_col,
            gen_answer_col=f"star_generation_answer_{iteration}",
            gen_rationale_col=f"star_generation_rationale_{iteration}",
            rat_answer_col=f"star_rationalization_answer_{iteration}",
            rat_rationale_col=f"star_rationalization_rationale_{iteration}",
            id_col=config.id_col
        )

        # Modify something in memory
        ft_config["data"]["dataset_name"] = fine_tuning_data_path
        generation_model_path = f"{model_checkpoint_dir}_{iteration}"
        ft_config["training"]["output_dir"] = generation_model_path
        ft_config['training']['run_name'] = f"{config.run_name}_{iteration}"

        updated_config_path = "configs/temp_STaR_ft_config.yaml"
        with open(updated_config_path, "w") as f:
            yaml.dump(config_dict, f, sort_keys=False)

        # (7) Fine-tune on the combined correct solutions
        call_fine_tune(
            config_yaml_path=updated_config_path,
            accelerate_config_path=args.accelerate_config_path
        )

        # End of iteration; M_{n} is now your updated model

    print("[INFO] STaR algorithm completed.")

if __name__ == "__main__":
    main()
