#!/bin/bash

# Function to generate training script for a given dataset
generate_script() {
    local dataset_name="$1"
    local script_name="${dataset_name}_100k.sh"
    local run_name="${dataset_name}_100k"
    
    echo "Generating script: ${script_name}"
    
    cat <<EOF > "scripts/${script_name}"
##### this is the exp name #####
run_name=${run_name}

# data mixture
dataset_names=(
    ${dataset_name}
)
num_trajectories=(
    100000
)

##### prepare data for the exp #####
# if the output directory already exists, remove it
if [ -d "/dev/shm/\${run_name}" ]; then
    rm -rf "/dev/shm/\${run_name}"
fi

# create the output directory
mkdir -p "/dev/shm/\${run_name}"

for i in "\${!dataset_names[@]}"; do
    dataset_name="\${dataset_names[i]}"
    num_traj="\${num_trajectories[i]}"
    input_dir="/home/clouduser/Code/data/gym/\${dataset_name}/train"
    output_dir="/dev/shm/\${run_name}"
    
    echo "Processing dataset: \${dataset_name} with \${num_traj} trajectories"
    
    # Check if input directory exists
    if [ ! -d "\${input_dir}" ]; then
        echo "Error: Input directory does not exist: \${input_dir}"
        exit 1
    fi
    
    python /home/clouduser/Code/Github/visgym_training/src/to_conversation.py \\
        --input-dir "\${input_dir}" \\
        --output-dir "\${output_dir}" \\
        --dataset-name "\${dataset_name}" \\
        --num-trajectories "\${num_traj}" \\
        --trajectories-per-batch 100 \\
        --remove-zero-reward \\
        --remove-inference-contamination \\
        --inference-hash-file "/home/clouduser/Code/Github/visgym_training/hashes_test_set_v10.json" \\
        --num-workers 0
    
    # Check if the conversion was successful
    if [ \$? -ne 0 ]; then
        echo "Error: Failed to convert dataset \${dataset_name}"
        exit 1
    fi
done

echo "Consolidating conversations..."
python /home/clouduser/Code/Github/visgym_training/src/consolidate_conversations.py \\
    --input-dir "/dev/shm/\${run_name}" \\
    --output-dir "/dev/shm/\${run_name}" \\
    --remove_shards_after_consolidation

# Check if consolidation was successful
if [ \$? -ne 0 ]; then
    echo "Error: Failed to consolidate conversations"
    exit 1
fi

##### create dataset_info.json for llama factory #####
echo "Creating dataset_info.json..."
cat <<EOJ > dataset_info.json
{
    "\${run_name}": {
        "file_name": "/dev/shm/\${run_name}/consolidated_conversations.jsonl",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images"
        }
    }
}
EOJ

# Check if consolidated file exists
if [ ! -f "/dev/shm/\${run_name}/consolidated_conversations.jsonl" ]; then
    echo "Error: Consolidated conversations file not found"
    exit 1
fi

mkdir -p "/dev/shm/data_info"
mv dataset_info.json "/dev/shm/data_info/"

# Check if dataset_info.json was created successfully
if [ ! -f "/dev/shm/data_info/dataset_info.json" ]; then
    echo "Error: Failed to create dataset_info.json"
    exit 1
fi

##### generate training config yaml file #####
echo "Generating training config YAML file..."
cat <<EOY > "/home/clouduser/Code/Github/visgym_training/configs/\${run_name}.yaml"
### model
model_name_or_path: /home/clouduser/Code/Models/Qwen2.5-VL-7B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: false
freeze_multi_modal_projector: false
freeze_language_model: false
deepspeed: /home/clouduser/Code/Github/visgym_training/configs/ds_z2_config.json

### dataset
dataset_dir: /dev/shm/data_info
dataset: \${run_name}
buffer_size: 256
preprocessing_batch_size: 256
streaming: true
accelerator_config:
  dispatch_batches: false
template: qwen2_vl
cutoff_len: 15000
max_steps: 1500
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /dev/shm/models/\${run_name}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
EOY

# Check if YAML config was created successfully
if [ ! -f "/home/clouduser/Code/Github/visgym_training/configs/\${run_name}.yaml" ]; then
    echo "Error: Failed to create training config YAML file"
    exit 1
fi

##### train the model #####
echo "Starting model training..."
llamafactory-cli train "/home/clouduser/Code/Github/visgym_training/configs/\${run_name}.yaml"

# Check if training was successful
if [ \$? -ne 0 ]; then
    echo "Error: Model training failed"
    exit 1
fi

echo "Training completed successfully!"
EOF

    # Make the generated script executable
    chmod +x "scripts/${script_name}"
    echo "Generated and made executable: scripts/${script_name}"
}

# Define all datasets from DATASETS6
ALL_DATASETS6=(
    # "colorization"
    # "counting/guess_only"
    # "counting/mark_all"
    # "jigsaw/reorder"
    # "jigsaw/swap"
    # "mental_rotation_2d"
    # "fetch_reach"
    # "refdot"
    # "sliding_block"
    # "toy_maze_2d"
    # "zoom_in_puzzle/reorder"
    # "zoom_in_puzzle/swap"
    # "fetch_pick_place"
    # "toy_maze_3d"
    # "patch_reassembly"
    # "matchstick_rotation"
    # "video_unshuffle/reorder"
    # "video_unshuffle/swap"
    # "mental_rotation_3d_cube"
    # "mental_rotation_3d_objaverse"
    # "matchstick_equation/bfs"
    # "matchstick_equation/dfs"
    # "matchstick_equation/sos"
)

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Main script logic
if [ $# -eq 0 ]; then
    echo "No arguments provided. Generating scripts for all DATASETS6..."
    echo "This will generate ${#ALL_DATASETS6[@]} scripts:"
    printf "  - %s_100k.sh\n" "${ALL_DATASETS6[@]}"
    echo ""
    
    # Generate scripts for all datasets
    for dataset_name in "${ALL_DATASETS6[@]}"; do
        generate_script "$dataset_name"
    done
    
elif [ "$1" == "--list" ]; then
    echo "Available datasets in DATASETS6:"
    printf "  - %s\n" "${ALL_DATASETS6[@]}"
    echo ""
    echo "Usage examples:"
    echo "  $0                                    # Generate all scripts"
    echo "  $0 colorization                   # Generate single script"
    echo "  $0 colorization counting/guess_only  # Generate multiple scripts"
    exit 0
    
else
    echo "Generating scripts for specified datasets..."
    # Generate scripts for provided dataset names
    for dataset_name in "$@"; do
        generate_script "$dataset_name"
    done
fi

echo ""
echo "All scripts generated successfully!"
echo "Generated scripts are located in: scripts/"
echo ""
echo "To run a specific script:"
echo "  cd scripts"
echo "  ./colorization_100k.sh"