# Paper Title

This repository contains code for paper ...

## Running STaSC

To reproduce the experiments and run any version of the algorithm described in paper. To run the algorithm you need to run the following with the specified configs. the configs description will be below.

```
 CUDA_VISIBLE_DEVICES=0,1 python star_correction.py --config configs/self_correction_star_config.yaml --ft_config configs/fine_tune.yaml --accelerate_config_path configs/accelerate_config.yaml 
```

## Self Correction Configuration

Here the self correction algorithm configuration is described

### 📌 Model Configuration
| Parameter                | Value                                      | Description |
|--------------------------|--------------------------------------------|-------------|
| `model_path`             | `Qwen/Qwen2.5-1.5B-Instruct`           | Pretrained model checkpoint path. |
| `cache_dir`             | `/home/cache/`       | Path to the cache directory for storing model weights. |
| `gpu_memory_utilization` | `0.9`                                    | GPU memory allocation fraction (0.0 - 1.0). |
| `enforce_eager`         | `True`                                    | Whether to enforce eager execution for debugging. |
| `max_model_len`         | `8192`                                   | Maximum sequence length for the model. |

### 📊 Dataset Configuration
| Parameter         | Value                       | Description |
|------------------|---------------------------|-------------|
| `data_path`      | `data/datasets/s_nq`       | Path to the dataset used for QA evaluation. |
| `id_col`         | `question_id`              | Unique identifier column in the dataset. |
| `question_col`   | `question_text`            | Column containing the input question text. |
| `gold_col`       | `reference`                | Column containing the reference (gold) answer. |

### 🔁 Self-Correction Algorithm Parameters
| Parameter                        | Value   | Description |
|----------------------------------|---------|-------------|
| `num_star_iterations`            | `10`     | Number of self-correction iterations. |
| `run_name`                       | `test`  | Name for the experiment run. |
| `initial_answer_with_new_model`  | `True`  | Whether to generate initial answers with \(M_0\) (True) or \(M_{n-1}\) (False). |
| `only_better_correction`         | `True`  | Whether to use strictly improving corrections (True) or allow neutral ones (False). |
| `train_from_initial_model`       | `True`  | Whether to fine-tune from \(M_0\) (True) or from \(M_{n-1}\) (False). |

### 📝 Generation Configuration
| Parameter                      | Value         | Description |
|--------------------------------|-------------|-------------|
| `few_shot_dir`                 | `few_shots` | Directory containing few-shot examples for prompting. |
| `number_output_initial_generations` | `5` | Number of initial generations per question. |
| `number_output_corrections`    | `5` | Number of correction generations per question. |

### 🎲 Sampling Parameters
| Parameter       | Value  | Description |
|---------------|--------|-------------|
| `temperature` | `0.6`  | Sampling temperature (higher = more randomness). |
| `top_p`       | `0.9`  | Nucleus (Top-p) sampling threshold. |
| `max_tokens`  | `256`  | Maximum number of tokens to generate. |

