# Scaling up Masked Diffusion Models on Text

[![arXiv](https://img.shields.io/badge/arXiv-2410.18514-red.svg)](https://arxiv.org/abs/2410.18514)
[![deploy](https://img.shields.io/badge/Huggingface%20-SMDM%20-blue)](https://huggingface.co/nieshen/SMDM)

Masked diffusion models (MDMs) have shown promise in language modeling, yet their scalability and effectiveness in core 
language tasks, such as text generation and language understanding, remain underexplored. This paper establishes the 
first scaling law for MDMs, demonstrating a scaling rate comparable to autoregressive models (ARMs) and a relatively 
small compute gap. Motivated by their scalability, we train a family of MDMs with up to 1.1 billion (B) parameters to 
systematically evaluate their performance against ARMs of comparable or larger sizes. Fully leveraging the probabilistic 
formulation of MDMs, we propose a simple yet effective *unsupervised classifier-free* guidance that effectively 
exploits large-scale unpaired data, boosting performance for conditional inference. In language understanding, the 
1.1B MDM outperforms the 1.1B TinyLlama model trained on the same data across four of eight zero-shot benchmarks. 
Notably, it achieves competitive math reasoning ability with the 7B Llama-2 model on the GSM8K dataset. In text 
generation, MDMs provide a flexible trade-off compared to ARMs utilizing KV-cache: MDMs match the performance of 
ARMs while being 1.4 times faster or achieving higher quality than ARMs at a higher computational cost. Moreover, 
MDMs address challenging tasks for ARMs by effectively handling bidirectional reasoning and adapting to temporal 
shifts in data. Notably, a 1.1B MDM breaks the *reverse curse* encountered by much larger ARMs with significantly 
more data and computation, such as 13B Llama-2 and 175B GPT-3.


<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/scale_loss.png" style="width: 48%; margin-right: 0.5%;" />
    <img src="./imgs/scale_para.png" style="width: 48%;"/>
</div>



## Dependency
The repository now includes a ready-to-use Conda environment file:

```sh
conda env create -f environment.yaml
conda activate smdm
```

This environment already includes the main packages used by the current codebase, including `torch`, `transformers`, `lightning`, `lm-eval`, `bitsandbytes`, `openai`, `fschat`, `anthropic`, and the math-correction utilities.

If you prefer to build from the original TinyLlama setup, you can still start from the [TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md) environment and install the extra dependencies used here. We also keep additional installation notes in [CONDA.md](CONDA.md).

## Pretrained models
We provided all pretrained models on [Huggingface](https://huggingface.co/nieshen/SMDM), including those 
for the scaling laws experiment, the conditional generation experiment, 
and the reverse curse experiment. 

We hope that the series of pretrained ARMs and MDMs will contribute to the advancement of the field.


## Pretrain
Please first use the code provided by [TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md) to preprocess the 
[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) dataset and the put the data chunks into `/dataset/slim_star_combined`.


### Pretrain ARMs
```sh
# e.g., 1028M non-embedding parameters ARM and 100e18 training FLOPs, 8 GPUs
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    pretrain/train_ar.py --model 1028 --flops 100.
```


### Pretrain MDMs
```sh
# e.g., 170M non-embedding parameters MDM and 10e18 training FLOPs, 8 GPUs
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    pretrain/train_mdm.py --model 170 --flops 10.
```


### Pretrain MDMs with stochastic sequence length
```sh
# e.g., 170M non-embedding parameters MDM and 60e18 training FLOPs, 8 GPUs
# set 1% data to a stochastic sequence length
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    pretrain/train_mdm_rl.py --model 170 --flops 60. --ssl_ratio 0.01
```

### Multi-machine training
```sh
# e.g., 1028M non-embedding parameters MDM and 1600e18 training FLOPs
# set 1% data to a stochastic sequence length
# 2 machines, 16 GPUs
lightning run model \
    --node-rank=$RANK  \
    --main-address=$MASTER_ADDR \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=2 \
    pretrain/train_mdm_rl.py --model 1028 --flops 1600. --ssl_ratio 0.01 --nodes_num 2
```

## Supervised fine-tuning
### Math reasoning
Please download the augmented training [data](https://github.com/da03/implicit_chain_of_thought/blob/main/data/gsm8k/train.txt) and
put the `train.txt` file in `./data/gsm8k`.
```angular2html
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    sft/finetune_mdm_gsm8k.py --model 1028 --pretrain_path models/mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors
```

### Conditional generation
Please download the [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json) dataset and put the json file in `./data`.
Following [CLLM](https://github.com/hao-ai-lab/Consistency_LLM), we only used the first round of dialogue data.
```sh
# Finetune ARMs
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    sft/finetune_ar.py --model 1028 --pretrain_path models/ar-1028M-100e18.safetensors
    
    
# Finetune MDMs
# For the unsupervised CFG, we set --cfg to 0.
# For the standard CFG, we set --cfg to 0.1
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    sft/finetune_mdm.py --model 1028 --pretrain_path models/mdm-1028M-1600e18.safetensors --cfg 0.
```

### Reverse curse
Please download the `reverse_experiments` folder provided by [lukasberglund](https://github.com/lukasberglund/reversal_curse/tree/main/data/reverse_experiments) and place it in `./data`.
```sh
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    sft/finetune_mdm_reverse.py --model 1028 --pretrain_path models/mdm-1028M-1600e18.safetensors
```

## Evaluation

### Commonsense reasoning and reading comprehension
We use the famous [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework for evaluation.


#### GPT-2
```sh
lm_eval --model hf \
    --model_args pretrained=openai-community/gpt2-xl,dtype="float" \
    --tasks hellaswag,openbookqa,arc_easy,boolq,piqa,social_iqa,race,lambada_standard \
    --device cuda:0
```

#### TinyLlama
We evaluate TinyLlama with 3.3e21 pre-training FLOPs.
```angular2html
lm_eval --model hf \
    --model_args pretrained=TinyLlama/tinyLLaMA-v1.1-checkpoints,revision=step-300000,dtype="bfloat16" \
    --tasks hellaswag,openbookqa,arc_easy,boolq,piqa,social_iqa,race,lambada_standard \
    --device cuda
```

#### ARMs pretrained on the SlimPajama dataset
```sh
python evaluate_ar.py --tasks hellaswag,openbookqa,arc_easy,boolq,piqa,social_iqa,race,lambada_standard --model ar --model_args model_name=170,ckpt_path='models/ar-170M-100e18.safetensors'
```

#### MDMs pretrained on the SlimPajama dataset
We provide the running commands in `eval_mdm.sh`.


### Math reasoning
Please download the GSM8K test [data](https://github.com/hao-ai-lab/Consistency_LLM/blob/main/eval/gsm8k/test.jsonl)
and put the `test.jsonl` into `./data/gsm8k`
```angular2html
python evaluate_gsm8k.py --ckpt_path "models/mdm-1028M-3300e18-rsl-gsm8k.safetensors"
```

### GSM8K correction / refinement
The current repository also includes a GSM8K correction pipeline for refining failed or corrupted solutions. The workflow is:

1. Generate buggy GSM8K solutions with injected number or symbol/operator errors.
2. Evaluate the buggy JSONL with `eval_gsm8k_jsonl.py` to mark `test_passed`.
3. Refine only failed samples with `refine_gsm8k.py`.
4. Re-evaluate the refined outputs.
5. Visualize the refined trajectories and remask behavior.

#### 1. Generate buggy GSM8K data
`mathcorrection/generate.py` creates corrupted GSM8K answers. The main modes are:

- `--error_type number --propagate_numbers` for number propagation errors
- `--error_type symbol` for operator / symbol errors

Example:
```sh
python mathcorrection/generate.py \
    --dataset gsm8k \
    --error_type number \
    --propagate_numbers \
    --model_name yiheng0824/smdm/latest.pth \
    --data_path mathcorrection \
    --data_num 1 \
    --n_replace 1 \
    --skip_existing
```

This produces files such as:
```text
mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1.jsonl
```

#### 2. Evaluate buggy samples
Evaluate the generated JSONL first so failed examples are marked with `test_passed=False`:

```sh
python eval_gsm8k_jsonl.py \
    --results_file mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1.jsonl \
    --dataset gsm8k \
    --initial_dataset mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1.jsonl
```

This writes:
```text
mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1_evaluated.jsonl
```

#### 3. Refine failed samples
The current refinement entry point is `refine_gsm8k.py`. By default it uses:

- `--refine_mode two_stage`
- `--refine_setting remove_all`
- `--sampler_backend llada`

Example:
```sh
torchrun --nproc_per_node=1 refine_gsm8k.py \
    --initial_results_file mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1_evaluated.jsonl \
    --model_name yiheng0824/smdm/latest.pth \
    --batch_size 1 \
    --refined_steps 2 \
    --algorithm self_conf-remask:vanilla \
    --temperature 0.0 \
    --refine_setting remove_all \
    --refine_mode two_stage
```

Important current options:

- `--sampler_backend llada|eval_diff`
- `--eval_diff_mode generate|edit`
- `--stage2_mode generate|edit`
- `--add_correction_instruction`
- `--skip_existing`

The refinement code saves both JSONL outputs and per-step histories. The output directory structure is:

```text
correction_results/refined_steps{N}/{refine_setting}/{algorithm_with_temp}/{input_dir}/{input_stem}/
correction_history/refined_steps{N}/{refine_setting}/{algorithm_with_temp}/{input_dir}/{input_stem}/
```

For the example above, the main refined file is:
```text
correction_results/refined_steps2/remove_all/self_conf-remask_vanilla_t00/mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1_evaluated/latest.pth_number_propagate_1_wrong_1_evaluated_results_refined.jsonl
```

#### 4. Evaluate refined outputs
Use the same evaluator on the refined JSONL:

```sh
python eval_gsm8k_jsonl.py \
    --results_file correction_results/refined_steps2/remove_all/self_conf-remask_vanilla_t00/mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1_evaluated/latest.pth_number_propagate_1_wrong_1_evaluated_results_refined.jsonl \
    --dataset gsm8k \
    --initial_dataset mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1_evaluated.jsonl
```

This produces a companion `_evaluated.jsonl` file with `test_passed` and `error_message`.

#### 5. Visualize refinement and remask trajectories
To visualize a mix of passed and failed refined samples:

```sh
python visualize_passed_refined.py \
    correction_results/refined_steps2/remove_all/self_conf-remask_vanilla_t00/mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1_evaluated/latest.pth_number_propagate_1_wrong_1_evaluated_results_refined_evaluated.jsonl \
    correction_history/refined_steps2/remove_all/self_conf-remask_vanilla_t00/mathcorrection/gsm8k/latest.pth_number_propagate_1_wrong_1_evaluated \
    --mode both \
    --model yiheng0824/smdm/latest.pth
```

This generates HTML files such as:

- `refine_diff_passed.html`
- `remask_passed.html`

You can also call the lower-level visualizers directly:

```sh
python visualize_remask.py <history_dir> --evaluated-jsonl <refined_evaluated_jsonl>
python visualize_refine_diff.py <refined_results_jsonl> <history_dir> --evaluated-jsonl <refined_evaluated_jsonl>
```

#### End-to-end test script
For the full GSM8K correction workflow, see:

```sh
bash test_refine_gsm8k.sh
```

This script runs buggy-data generation, evaluation, refinement, refined evaluation, and visualization for a list of models and refinement step counts.


### Conditional generation
We measure the MT-Bench score using the [fast-chat](https://github.com/lm-sys/FastChat) framework. We first generate model responses and put the responses in the json files.
```angular2html
# ARMs
python eval/gen_model_answer.py --model-id 1028 --model-type 'arm' --model-path "models/ar-1028M-100e18-sharegpt.safetensors" --answer-file "data/mt_bench/model_answer/arm.jsonl" 

# MDMs
python eval/gen_model_answer.py --model-id 1028 --model-type 'mdm' --model-path "models/mdm-1028M-1600e18-sharegpt.safetensors" --steps 128 --cfg-scale 0.6 --answer-file "data/mt_bench/model_answer/mdm.jsonl" 
```

Then we use GPT-4o to score.
```angular2html
export OPENAI_API_KEY=xxxxxxxxx
python eval/gen_judgment.py  --parallel 10 --judge-model "gpt-4o-2024-05-13"
python eval/show_result.py  --judge-model "gpt-4o-2024-05-13"
```

### Reverse curse
```angular2html
# NameToDescription
python evaluate_reverse.py --qs_type ntd --model 1028 --ckpt-path "models/mdm-1028M-1600e18-reverse.safetensors"

# DescriptionToName
python evaluate_reverse.py --qs_type dtn --model 1028 --ckpt-path "models/mdm-1028M-1600e18-reverse.safetensors"
```

### Temporal quality degradation
We first preprocess the [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset. Due to version conflicts, we need to create a new Anaconda environment to preprocess the FineWeb dataset.
```angular2html
conda create -n fineweb python=3.10
conda activate fineweb

pip install datatrove==0.2.0 transformers pyarrow
```

Then preprocess the Fineweb dataset.
```angular2html
python scripts/prepare_fineweb.py
```

Evaluate ARMs and MDMs on the Fineweb data.
```angular2html
# "CC-MAIN-2024-18": April 2024, "CC-MAIN-2024-10": February/March 2024

# ARMs
python evaluate_fineweb.py --type arm --model 170  --ckpt-path 'models/ar-170M-6e18.safetensors' --fineweb "CC-MAIN-2024-10"

# MDMs. To improve speed, the number of Monte Carlo estimations can be reduced, for example, down to 16.
python evaluate_fineweb.py --type mdm --model 170  --ckpt-path 'models/mdm-170M-100e18.safetensors' --fineweb "CC-MAIN-2024-18" --mc-samples 128
```