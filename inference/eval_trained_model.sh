#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# VLM-Gym Trained Model Evaluation Script
# ============================================================================
# Example usage:
#   CUDA_VISIBLE_DEVICES=0 ./eval_trained_model.sh
#   MODEL=mixed_qwen3vl SPLIT=hard ./eval_trained_model.sh
# ============================================================================

# -------- Configuration --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model and task configuration (can override via environment)
: "${MODEL:=mixed_qwen3vl}"  # update to the model you want to evaluate
: "${SPLIT:=easy}"           # easy or hard
: "${TASK:=maze_3d}"          # task name without split suffix

# Paths
: "${MODEL_ROOT:=${SCRIPT_DIR}/../ckpts}"
: "${DATASET_ROOT:=${SCRIPT_DIR}/inference_dataset}"
: "${OUT_ROOT:=${SCRIPT_DIR}/eval_trained}"

# -------- Validate paths --------
MODEL_PATH="${MODEL_ROOT}/${MODEL}"
EPISODES_ROOT="${DATASET_ROOT}/test_set_${SPLIT}"
INITDIR="${DATASET_ROOT}/initial_states_${SPLIT}"
OUTDIR="${OUT_ROOT}/${MODEL}_${TASK}_${SPLIT}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] Model path not found: $MODEL_PATH"
  echo "Please set MODEL_ROOT or download the model checkpoint"
  exit 1
fi

if [[ ! -d "$EPISODES_ROOT" ]]; then
  echo "[ERROR] Episodes root not found: $EPISODES_ROOT"
  echo "Please run download_data.sh first or set DATASET_ROOT"
  exit 1
fi

if [[ ! -d "$INITDIR" ]]; then
  echo "[ERROR] Initial states dir not found: $INITDIR"
  echo "Please run download_data.sh first or set DATASET_ROOT"
  exit 1
fi

# -------- Set max steps based on split --------
MAX_STEPS=20
[[ "$SPLIT" == "hard" ]] && MAX_STEPS=30

# -------- Create output directory --------
mkdir -p "${OUTDIR}"

# -------- Run inference --------
echo "========================================"
echo "Evaluating Trained Model"
echo "  Model: ${MODEL}"
echo "  Task: ${TASK}"
echo "  Split: ${SPLIT}"
echo "  Model Path: ${MODEL_PATH}"
echo "  Episodes: ${EPISODES_ROOT}"
echo "  InitDir: ${INITDIR}"
echo "  Output: ${OUTDIR}"
echo "  Max Steps: ${MAX_STEPS}"
echo "========================================"

python "${SCRIPT_DIR}/run_inference.py" \
    --model "${MODEL_PATH}" \
    --episodes_root "${EPISODES_ROOT}" \
    --initdir "${INITDIR}" \
    --api_option transformers \
    --outdir "${OUTDIR}" \
    --make-gif \
    --elicit-verbalization \
    --include-env-feedback \
    --max-steps "${MAX_STEPS}"

echo ""
echo "Evaluation complete!"
echo "Results saved to: ${OUTDIR}"