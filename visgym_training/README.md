# Launch Scripts

This repository contains scripts to launch training jobs for [VisGym](visgym.github.io) locally or using Docker.

## Prerequisites

*   Python 3.8+
*   Docker (optional, for containerized execution)
*   Git LFS (for data download)

## Setup

1.  **Download Data**:
    Run the data transfer script to download the datasets from Hugging Face.
    ```bash
    bash data_transfer.sh
    ```
    This will download the data to `/home/clouduser/Code/data/gym` (or `./data/gym` if running locally).

2.  **Install Dependencies** (if running locally without Docker):
    Ensure you have the necessary python packages installed.
    ```bash
    pip install -r requirements.txt # (If you have one, otherwise install manually)
    ```

3.  **Download Checkpoints**:
    Download the following checkpoints and place them in the `models/` directory:
    *   [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
    *   [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

## Usage

### Running Locally

To run a specific training script locally:

```bash
python launch_local.py --run_script_path ./scripts/your_script_name.sh
```

**Note**: The scripts contain absolute paths (e.g., `/home/clouduser/...`). `launch_local.py` automatically detects this when running locally and temporarily replaces them with your current directory paths (`./` and `./data/gym`). It assumes data is located in `./data/gym` and models in `./models`.

### Running with Docker

To run the script inside a Docker container:

```bash
python launch_local.py --run_script_path ./scripts/your_script_name.sh --docker
```

This will automatically build the Docker image from the `Dockerfile` in the current directory (tagging it as `launch_gym_local`) and run the script inside it. It mounts the current directory to `/home/clouduser/Code/Github/visgym_training` and attempts to mount `../../data/gym` to `/home/clouduser/Code/data/gym`.

You can specify a different image or GPUs:

```bash
python launch_local.py --run_script_path ./scripts/your_script_name.sh --docker --image your/image:tag --gpus 0,1
```

### Batch Submission

To run multiple scripts defined in `submmit_all.sh` or `submmit_all_mixed.sh`:

```bash
bash submmit_all.sh
```

or

```bash
bash submmit_all_mixed.sh
```

## Directory Structure

*   `launch_local.py`: Main launcher script.
*   `data_transfer.sh`: Script to download datasets.
*   `scripts/`: Directory containing training shell scripts.
*   `scripts_mixed/`: Directory containing mixed training shell scripts.
*   `configs/`: Configuration files.
*   `src/`: Source code for data processing.


