#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Array of all script names to be executed
scripts=(
    # "mixed_all_v7_300k.sh"
    # "mixed_all_v7_300k_freeze_lang.sh"
    # "mixed_all_v7_300k_freeze_vision.sh"
    # "mixed_all_v7_300k_qwen3.sh"
    # "mixed_all_v7_300k_qwen3_freeze_language.sh"
    # "mixed_all_v7_300k_qwen3_freeze_vision.sh"
)

# Loop through each script in the array and launch it
for script_name in "${scripts[@]}"; do
    echo "=================================================="
    echo "==> Launching script: ${script_name}"
    echo "=================================================="
    python launch_local.py --run_script_path "./scripts_mixed/${script_name}"
done

echo "### All launch commands have been executed. ###"