#!/bin/bash
# Test script for GSM8K refinement (number with propagate, or symbol/operator errors)
# Exit on error, undefined variables, and pipe failures
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

# Base directory for generating buggy data
DATA_PATH_BASE="mathcorrection"

# Test configuration - use small values for testing
# Args: N_REPLACE MODEL_NAME ALGORITHM VARIANTS_PER_SAMPLE ERROR_TYPE
# ERROR_TYPE: number (with propagate) | symbol | operator (alias for symbol)
N_REPLACE="${1:-1}"
MODEL_NAME="${2:-yiheng0824/smdm/cdlm_07.pth}"  # Default to cdlm_model.pth
ALGORITHM="${3:-self_conf-remask:vanilla}"
VARIANTS_PER_SAMPLE="${4:-1}"  # Default variants per sample
ERROR_TYPE_RAW="${5:-number}"
# Normalize: operator -> symbol for generate.py
if [[ "$ERROR_TYPE_RAW" == "operator" ]]; then
    ERROR_TYPE="symbol"
else
    ERROR_TYPE="${ERROR_TYPE_RAW}"
fi
if [[ "$ERROR_TYPE" != "number" && "$ERROR_TYPE" != "symbol" ]]; then
    echo "ERROR: ERROR_TYPE must be 'number', 'symbol', or 'operator' (operator=symbol). Got: ${ERROR_TYPE_RAW}"
    exit 1
fi

MODEL_DISPLAY_NAME=$(basename "$MODEL_NAME")

echo "=========================================="
echo "Testing GSM8K Refinement (error_type=${ERROR_TYPE_RAW})"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Algorithm: ${ALGORITHM}"
echo "N_replace: ${N_REPLACE}"
echo "Error type: ${ERROR_TYPE} (raw: ${ERROR_TYPE_RAW})"
echo "=========================================="

# Model-specific configuration (utils.py loads TransEncoder for .pth/smdm/cdlm)
if [[ "$MODEL_NAME" == *".pth" ]] || [[ "$MODEL_NAME" == *"smdm"* ]] || [[ "$MODEL_NAME" == *"cdlm"* ]]; then
    # Diffusion model (evaluate_gsm8k-style): TransEncoder + TinyLlama, mask_id=32000
    TEMP=0.0
elif [[ "$MODEL_NAME" == *"LLaDA-8B-Instruct"* ]]; then
    USE_CHAT_TEMPLATE="--use_chat_template"
    TEMP=0.0
elif [[ "$MODEL_NAME" == *"LLaDA-8B-Base"* ]]; then
    USE_CHAT_TEMPLATE="--no_chat_template"
    TEMP=0.0
elif [[ "$MODEL_NAME" == *"Dream-v0-Base-7B"* ]]; then
    USE_CHAT_TEMPLATE="--no_chat_template"
    TEMP=0.6
elif [[ "$MODEL_NAME" == *"Dream-v0-Instruct-7B"* ]]; then
    USE_CHAT_TEMPLATE="--use_chat_template"
    TEMP=0.6
elif [[ "$MODEL_NAME" == *"open-dcoder"* ]]; then
    USE_CHAT_TEMPLATE="--no_chat_template"
    TEMP=0.6
else
    USE_CHAT_TEMPLATE="--no_chat_template"
    TEMP=0.0
fi

DATASET="gsm8k"

echo "=========================================="
echo "Dataset: ${DATASET}, Error type: ${ERROR_TYPE}"
echo "=========================================="

# (1) Generate buggy data
# Path format matches generate.py:
#   number + propagate: {model}_number_propagate_{variants}_wrong_{n_replace}.jsonl
#   symbol/operator:    {model}_symbol_{variants}_wrong_{n_replace}.jsonl
BUGGY_DIR="${DATA_PATH_BASE}/${DATASET}"
mkdir -p "${BUGGY_DIR}"
if [[ "$ERROR_TYPE" == "number" ]]; then
    INITIAL_RESULTS_FILE="${BUGGY_DIR}/${MODEL_DISPLAY_NAME}_${ERROR_TYPE}_propagate_${VARIANTS_PER_SAMPLE}_wrong_${N_REPLACE}.jsonl"
else
    INITIAL_RESULTS_FILE="${BUGGY_DIR}/${MODEL_DISPLAY_NAME}_${ERROR_TYPE}_${VARIANTS_PER_SAMPLE}_wrong_${N_REPLACE}.jsonl"
fi

if [ ! -f "${INITIAL_RESULTS_FILE}" ]; then
    echo "Generating buggy data: ${INITIAL_RESULTS_FILE}"
    GEN_ARGS=(
        --dataset "${DATASET}"
        --error_type "${ERROR_TYPE}"
        --model_name "${MODEL_NAME}"
        --data_path "${DATA_PATH_BASE}"
        --data_num "${VARIANTS_PER_SAMPLE}"
        --n_replace "${N_REPLACE}"
        --skip_existing
    )
    if [[ "$ERROR_TYPE" == "number" ]]; then
        GEN_ARGS+=(--propagate_numbers)
    fi
    python mathcorrection/generate.py "${GEN_ARGS[@]}"
else
    echo "Buggy data already exists: ${INITIAL_RESULTS_FILE}"
fi

# Evaluate initial buggy data to get failed samples for refinement
INITIAL_EVALUATED_FILE="${INITIAL_RESULTS_FILE%.jsonl}_evaluated.jsonl"
echo "Evaluating initial buggy data: ${INITIAL_RESULTS_FILE}"
python eval_gsm8k_jsonl.py \
    --results_file "${INITIAL_RESULTS_FILE}" \
    --dataset "${DATASET}" \
    --initial_dataset "${INITIAL_RESULTS_FILE}" \
    --skip_if_exist

# Use evaluated file for refinement (contains test_passed field)
INITIAL_RESULTS_FILE="${INITIAL_EVALUATED_FILE}"

# Test with only a few refinement steps for quick testing
REFINED_STEPS_LIST=(4)
REFINE_SETTING="remove_all"
ALGORITHM_SAFE=$(echo "$ALGORITHM" | sed 's/:/_/g')
MASTER_PORT=29500

for refined_steps in "${REFINED_STEPS_LIST[@]}"; do
    echo "=========================================="
    echo "Running refinement with steps=${refined_steps}"
    echo "=========================================="

    REFINE_PORT=$((MASTER_PORT + refined_steps))
    echo "Using master port: ${REFINE_PORT}"

    # Run refinement
    torchrun --nproc_per_node=1 --master_port=${REFINE_PORT} refine_gsm8k.py \
        --initial_results_file "${INITIAL_RESULTS_FILE}" \
        --model_name "${MODEL_NAME}" \
        --batch_size 1 \
        --refined_steps "${refined_steps}" \
        --algorithm "${ALGORITHM}" \
        --temperature ${TEMP} \
        --refine_setting "${REFINE_SETTING}"
    # torchrun --nproc_per_node=1 --master_port=${REFINE_PORT} refine_gsm8k.py \
    #     --initial_results_file "${INITIAL_RESULTS_FILE}" \
    #     --model_name "${MODEL_NAME}" \
    #     --batch_size 8 \
    #     --refined_steps "${refined_steps}" \
    #     --sampler_backend eval_diff \
    #     --eval_cfg1 0.1 \
    #     --eval_cfg2 0.1 \
    #     --eval_context_length 256

    # Path logic must match refine_gsm8k.py build_output_paths (utils.py)
    # Output: correction_results/refined_steps{N}/{setting}/{algorithm_suffix}/{input_dir}/{input_stem}/{input_stem}_results_refined.jsonl
    # Algorithm suffix includes temperature (e.g. self_conf-remask_vanilla_t00 for temp=0.0)
    # Match utils.py: temp_str = f"{temperature:.1f}".replace('.', '')  e.g. 0.0->00, 0.6->06
    TEMP_STR=$(printf '%.1f' "${TEMP}" | tr -d '.')
    ALGORITHM_SUFFIX="${ALGORITHM_SAFE}_t${TEMP_STR}"
    INPUT_DIR=$(dirname "${INITIAL_RESULTS_FILE}")
    INPUT_STEM=$(basename "${INITIAL_RESULTS_FILE}" .jsonl)
    REFINED_RESULTS_DIR="correction_results/refined_steps${refined_steps}/${REFINE_SETTING}/${ALGORITHM_SUFFIX}/${INPUT_DIR}/${INPUT_STEM}"
    REFINED_RESULTS_FILE="${REFINED_RESULTS_DIR}/${INPUT_STEM}_results_refined.jsonl"
    REFINED_HISTORY_DIR="correction_history/refined_steps${refined_steps}/${REFINE_SETTING}/${ALGORITHM_SUFFIX}/${INPUT_DIR}/${INPUT_STEM}"
    
    # (3) Evaluate refined results
    if [ -f "${REFINED_RESULTS_FILE}" ]; then
        echo "Evaluating refined results..."
        python eval_gsm8k_jsonl.py \
            --results_file "${REFINED_RESULTS_FILE}" \
            --dataset "${DATASET}" \
            --initial_dataset "${INITIAL_RESULTS_FILE}"
        
        # (4) Visualize passed refined samples
        REFINED_EVALUATED_FILE="${REFINED_RESULTS_DIR}/${INPUT_STEM}_results_refined_evaluated.jsonl"
        if [ -f "${REFINED_EVALUATED_FILE}" ]; then
            echo "Visualizing passed refined samples..."
            python visualize_passed_refined.py \
                "${REFINED_EVALUATED_FILE}" \
                "${REFINED_HISTORY_DIR}" \
                --mode both \
                --model "${MODEL_NAME}"
        else
            echo "Warning: ${REFINED_EVALUATED_FILE} does not exist, skipping visualization"
        fi
    else
        echo "Warning: ${REFINED_RESULTS_FILE} does not exist, skipping evaluation"
    fi

    echo "Completed steps=${refined_steps}"
    echo ""
done

echo "=========================================="
echo "Test complete!"
echo "=========================================="
