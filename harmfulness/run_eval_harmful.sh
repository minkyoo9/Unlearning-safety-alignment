#!/usr/bin/env zsh

# Set the directory variables
SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

# Default values
EVAL_DATASET="AdvBench" # Default evaluation dataset
MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-3B-Instruct" # Default model path
BATCH_SIZE=128
OUTPUT_DIR="./outs_llama/out_${EVAL_DATASET}"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    arg="$1"
    shift
    case "${arg}" in
        --batch_size)
            BATCH_SIZE="$1"
            shift
            ;;
        --eval_dataset)
            EVAL_DATASET="$1"
            shift
            ;;
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$1"
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$1"
            shift
            ;;
        *)
            echo "Unknown parameter passed: '${arg}'" >&2
            exit 1
            ;;
    esac
done


mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
cp -f "$0" "${OUTPUT_DIR}/script.sh"

# Log output
exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

# Run inference
accelerate launch --config_file inference_config.yaml \
generate.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --expert_path "${EXPERT_PATH}" \
    --eval_dataset "${EVAL_DATASET}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${OUTPUT_DIR}"
echo "Generation Done"

# Run evaluation
python eval_llama_guard.py \
    --results_path "${OUTPUT_DIR}"
echo "Evaluation Done"
