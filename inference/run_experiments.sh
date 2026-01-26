#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# VLM-Gym Inference Experiment Runner
# ============================================================================
# Example usage:
#   ./run_experiments.sh                    # Run all models on all splits
#   SPLITS="easy" ./run_experiments.sh      # Run only easy split
#   MODELS="qwen3" ./run_experiments.sh     # Run only qwen3 model
# ============================================================================

: "${MAX_JOBS:=1}"

cleanup() {
  echo -e "\n[cleanup] Caught signal, terminating children..."
  jobs -pr | xargs -r kill -TERM 2>/dev/null || true
  sleep 2
  jobs -pr | xargs -r kill -KILL 2>/dev/null || true
}
trap cleanup INT TERM

# -------- Configuration --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTROOT="${SCRIPT_DIR}/eval"
DATASET_ROOT="${SCRIPT_DIR}/inference_dataset"

# Splits to run (can override via environment: SPLITS="easy" ./run_experiments.sh)
: "${SPLITS:=easy hard}"

# Models to run (can override via environment)
# Format: "model_name:api_option"
declare -A MODEL_CONFIGS=(
  # OpenAI
  ["gpt5"]="gpt-5:openai"
  # OpenRouter models
  ["gemini"]="google/gemini-2.5-pro:openrouter"
  ["qwen3"]="qwen/qwen3-vl-235b-a22b-instruct:openrouter"
  ["qwen"]="qwen/qwen2.5-vl-72b-instruct:openrouter"
  ["qwenvlmax"]="qwen/qwen-vl-max:openrouter"
  ["claude"]="anthropic/claude-sonnet-4:openrouter"
  ["grok"]="x-ai/grok-4-fast:openrouter"
  ["glm"]="z-ai/glm-4.5v:openrouter"
  ["intern"]="opengvlab/internvl3-78b:openrouter"
  ["llama"]="meta-llama/llama-4-maverick:openrouter"
  ["gemma"]="google/gemma-3-27b-it:openrouter"
  ["mistral"]="mistralai/mistral-medium-3.1:openrouter"
  ["uitars"]="bytedance/ui-tars-1.5-7b:openrouter"
  ["gemini3"]="google/gemini-3-pro-preview:openrouter"
)

# Which models to run (can override via environment: MODELS="qwen3 gpt5")
: "${MODELS:=qwen3}"

# Additional flags
MAKE_GIF="--make-gif"
EXTRA_FLAGS="--include-env-feedback"

# ------------------------

mkdir -p "${OUTROOT}"

run_experiment() {
  local model_tag="$1"
  local split="$2"

  # Parse model config
  local config="${MODEL_CONFIGS[$model_tag]}"
  if [[ -z "$config" ]]; then
    echo "[ERROR] Unknown model tag: $model_tag"
    echo "Available: ${!MODEL_CONFIGS[*]}"
    return 1
  fi

  local model_name="${config%%:*}"
  local api_option="${config##*:}"

  # Set paths based on split
  local episodes_root="${DATASET_ROOT}/test_set_${split}"
  local initdir="${DATASET_ROOT}/initial_states_${split}"
  local outdir="${OUTROOT}/${model_tag}_${split}"

  # Verify paths exist
  if [[ ! -d "$episodes_root" ]]; then
    echo "[ERROR] Episodes root not found: $episodes_root"
    return 1
  fi
  if [[ ! -d "$initdir" ]]; then
    echo "[ERROR] Initial states dir not found: $initdir"
    return 1
  fi

  mkdir -p "${outdir}"

  local max_steps_flag=()
  [[ "$split" == "hard" ]] && max_steps_flag=(--max-steps 30)
  
  echo "========================================"
  echo "Running: ${model_tag} on ${split}"
  echo "  Model: ${model_name}"
  echo "  API: ${api_option}"
  echo "  Episodes: ${episodes_root}"
  echo "  InitDir: ${initdir}"
  echo "  Output: ${outdir}"
  echo "========================================"

  python "${SCRIPT_DIR}/run_inference.py" \
    --model "${model_name}" \
    --api_option "${api_option}" \
    --episodes_root "${episodes_root}" \
    --initdir "${initdir}" \
    --outdir "${outdir}" \
    ${MAKE_GIF} \
    ${EXTRA_FLAGS} \
    "${max_steps_flag[@]}"
}

# Main loop
for model_tag in ${MODELS}; do
  for split in ${SPLITS}; do
    run_experiment "$model_tag" "$split"
  done
done

echo ""
echo "All experiments finished."
echo "Results saved to: ${OUTROOT}"
