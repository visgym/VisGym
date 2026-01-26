##### this is the exp name #####
run_name=match_equation-mixed_100k

# data mixture
dataset_names=(
    matchstick_equation/bfs
    matchstick_equation/dfs
    matchstick_equation/sos
)
num_trajectories=(
    40000
    40000
    40000
)

##### prepare data for the exp #####
# if the output directory already exists, remove it
if [ -d "/dev/shm/${run_name}" ]; then
    rm -rf "/dev/shm/${run_name}"
fi

# create the output directory
mkdir -p "/dev/shm/${run_name}"

for i in "${!dataset_names[@]}"; do
    dataset_name="${dataset_names[i]}"
    num_traj="${num_trajectories[i]}"
    input_dir="/home/clouduser/Code/data/gym/${dataset_name}/train"
    output_dir="/dev/shm/${run_name}"
    
    echo "Processing dataset: ${dataset_name} with ${num_traj} trajectories"
    
    # Check if input directory exists
    if [ ! -d "${input_dir}" ]; then
        echo "Error: Input directory does not exist: ${input_dir}"
        exit 1
    fi
    
    python /home/clouduser/Code/Github/visgym_training/src/to_conversation.py \
        --input-dir "${input_dir}" \
        --output-dir "${output_dir}" \
        --dataset-name "${dataset_name}" \
        --num-trajectories "${num_traj}" \
        --trajectories-per-batch 100 \
        --remove-zero-reward \
        --remove-inference-contamination \
        --inference-hash-file "/home/clouduser/Code/Github/visgym_training/hashes_test_set_v10.json" \
        --num-workers 0
    
    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to convert dataset ${dataset_name}"
        exit 1
    fi
done

echo "Consolidating conversations..."
python /home/clouduser/Code/Github/visgym_training/src/consolidate_conversations.py \
    --input-dir "/dev/shm/${run_name}" \
    --output-dir "/dev/shm/${run_name}" \
    --remove_shards_after_consolidation

# Check if consolidation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to consolidate conversations"
    exit 1
fi

##### create dataset_info.json for llama factory #####
echo "Creating dataset_info.json..."
cat <<EOJ > dataset_info.json
{
    "${run_name}": {
        "file_name": "/dev/shm/${run_name}/consolidated_conversations.jsonl",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images"
        }
    }
}
EOJ

# Check if consolidated file exists
if [ ! -f "/dev/shm/${run_name}/consolidated_conversations.jsonl" ]; then
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
cat <<EOY > "/home/clouduser/Code/Github/visgym_training/configs/${run_name}.yaml"
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
dataset: ${run_name}
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
output_dir: /dev/shm/models/${run_name}
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
if [ ! -f "/home/clouduser/Code/Github/visgym_training/configs/${run_name}.yaml" ]; then
    echo "Error: Failed to create training config YAML file"
    exit 1
fi

##### train the model #####
echo "Starting model training..."
llamafactory-cli train "/home/clouduser/Code/Github/visgym_training/configs/${run_name}.yaml"

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Model training failed"
    exit 1
fi

echo "Training completed successfully!"
