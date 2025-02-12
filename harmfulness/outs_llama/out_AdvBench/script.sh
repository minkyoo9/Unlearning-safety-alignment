#!/usr/bin/env zsh

# Set the directory variables
SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"


EVAL_DATASET="AdvBench" # AdvBench for crafting rejection responses / HEx-PHI & LLM-LAT_tot for evaluation
BATCH_SIZE=128

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    arg="$1"
    shift
    case "${arg}" in
        --expert_path)
            EXPERT_PATH="$1"
            shift
            ;;
        --num_samples)
            NUM_SAMPLES="$1"
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$1"
            shift
            ;;
        *)
            echo "Unknown parameter passed: '${arg}'" >&2
            exit 1
            ;;
    esac
done

# Loop through each model path and corresponding output directory
MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR="./outs_llama/out_${EVAL_DATASET}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
cp -f "$0" "${OUTPUT_DIR}/script.sh"

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

accelerate launch --config_file inference_config.yaml \
generate.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --expert_path "${EXPERT_PATH}" \
    --eval_dataset "${EVAL_DATASET}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${OUTPUT_DIR}"
echo "Generation Done"
python eval_llama_guard.py \
    --results_path "${OUTPUT_DIR}"
echo "Evaluation Done"