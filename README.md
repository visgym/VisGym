# VisGym
**Diverse, Customizable, Scalable Environments for Multimodal Agents**

VisGym is a gymnasium of **17 visually interactive, long-horizon environments** for evaluating, diagnosing, and training vision–language models (VLMs) in **multi-step visual decision-making** across symbolic puzzles, real-image understanding, navigation, and manipulation.

<p align="center">
  <video src="https://github.com/user-attachments/assets/6aea2ded-b15d-428b-81b8-24889c068240" width="100%" controls></video>
</p>


<p align="center">
  <a href="https://visgym.github.io/">🌐 Webpage</a> •
  <a href="https://arxiv.org/abs/2601.16973">🎓 arXiv</a> •
  <a href="https://huggingface.co/VisGym">🤗 Datasets/Models/Checkpoints</a>
</p>



## Contents
- [Get started](#get-started)
  - [Environment](#environment)
  - [Notebook playground](#notebook-playground)
- [Eval](#eval)
  - [1) Download data](#1-download-data)
  - [2) Run experiments](#2-run-experiments)
  - [3) Ablations](#3-ablations)
- [Training](#training)
- [Our models](#our-models)
  - [1) Released checkpoints](#1-released-checkpoints)
  - [2) Evaluate our model](#2-evaluate-our-model)
- [Citation](#citation)



# Get started

## Environment

### Prerequisites
- Python 3.11
- CUDA 12.1 (for GPU acceleration)
- Conda (recommended)

### (1) Create conda environment
```bash
conda create -n visgym python=3.11 -y
conda activate visgym
```

### (2) Install PyTorch with CUDA
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### (3) Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

### (4) Install VisGym (editable)
```bash
# Install gymnasium MuJoCo Fetch environments
pip install -e Gymnasium-Robotics/
pip install -e . 
pip install PyOpenGL==3.1.7 PyOpenGL-accelerate==3.1.7 --force-reinstall
# please ignore the dependency errors
```

### (5) Verify installation
```bash
python -c "import gymnasium; print('gymnasium import ok')"
python -c "import gymnasium_robotics; print('gymnasium_robotics import ok')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

> **Notes**
> - MuJoCo environments (Fetch Pick-and-Place, Fetch Reach) require the `gymnasium-robotics` package.
> - For CPU-only installation, skip step (2) and install PyTorch without CUDA: `pip install torch torchvision torchaudio`
> - Some 3D environments require OpenGL. On headless servers, you may need: `apt-get install -y libgl1-mesa-glx libosmesa6`
> - **MuJoCo rendering on headless servers**: Set the rendering backend before running:
>   ```bash
>   export MUJOCO_GL=egl   # For EGL (GPU rendering)
>   # or
>   export MUJOCO_GL=osmesa  # For OSMesa (CPU rendering)
>   ```

## Notebook playground
We recommend starting from the notebooks to:
- render a single environment rollout,
- inspect observations + feedback,
- test a baseline solver trajectory,
- debug action parsing / formatting,
- different difficulty levels.

Please see [gymnasium/demos/maze_2d.ipynb](gymnasium/demos/maze_2d.ipynb) for an example.



# Eval

VisGym evaluation is **multi-turn**: at each step the model receives the task instruction and the full interaction history of `(observation, action, feedback)` tuples, and outputs the next action.

Default paper-style evaluation settings:
- **Easy**: max **20** steps
- **Hard**: max **30** steps
- **70 episodes per task per setting**

## (1) Download data

VisGym typically separates evaluation artifacts into:
- **Partial dataset**: the minimal assets needed to run evaluation episodes (e.g., images/videos/initial states)
- **Eval metadata**: episode lists, seeds, split files, and environment configuration

Please see data structures and download from [VisGym/inference-dataset](https://huggingface.co/datasets/VisGym/inference-dataset).

We provide a script to download and organize the data:

```bash
# Install huggingface-cli
pip install -U "huggingface_hub[cli]"

# Download the dataset to local
# This will download 'assets/' and 'metadata/' folder into local dir
mkdir -p inference_dataset
huggingface-cli download VisGym/inference-dataset --repo-type dataset --local-dir ./inference_dataset
```

Training and development: 
- **Full dataset**: if you want to download the full dataset for training and development, please use `download_data.sh`. This allows you to generate episodes yourself for extension.

## (2) Run experiments

We provide `run_experiments.sh` to run evaluation across models and splits. The script supports both proprietary models (via OpenRouter/OpenAI) and open-weight models.

### Setup API keys
```bash
# For OpenRouter models (Claude, Gemini, Qwen, etc.)
export OPENROUTER_API_KEY="YOUR_KEY"

# For OpenAI models (GPT-5)
export OPENAI_API_KEY="YOUR_KEY"
```

### Run experiments
```bash
cd inference

# Run default model (qwen3) on all splits
./run_experiments.sh

# Run specific model on specific split
MODELS="gpt5" SPLITS="easy" ./run_experiments.sh

# Run multiple models
MODELS="qwen3 gemini claude" SPLITS="easy hard" ./run_experiments.sh
```

### Available models
The script supports the following model tags:
| Tag | Model | API |
|-----|-------|-----|
| `gpt5` | GPT-5 | OpenAI |
| `gemini` | Gemini 2.5 Pro | OpenRouter |
| `qwen3` | Qwen3-VL-235B | OpenRouter |
| `qwen` | Qwen2.5-VL-72B | OpenRouter |
| `claude` | Claude Sonnet 4 | OpenRouter |
| `grok` | Grok 4 Fast | OpenRouter |
| `llama` | LLaMA 4 Maverick | OpenRouter |
| `intern` | InternVL3-78B | OpenRouter |
| `glm` | GLM-4.5V | OpenRouter |
| `gemma` | Gemma 3-27B | OpenRouter |
| `mistral` | Mistral Medium 3.1 | OpenRouter |

### Output
Results are saved to `inference/eval/<model>_<split>/` with per-episode JSONL logs and trajectory GIFs.

## (3) Ablations

We provide additional scripts for ablation experiments:

### Text-only mode (no images)
Evaluates models using text descriptions instead of visual observations:
```bash
cd inference
MODELS="qwen3" ./run_experiments_text_mode.sh
```

### Final goal provided
Provides the final goal state image to the model:
```bash
cd inference
MODELS="qwen3" ./run_experiments_final_goal.sh
```

Both ablation scripts support the same `MODELS` and `SPLITS` environment variables as the main script.



# Training

VisGym supports SFT by generating demonstration trajectories from built-in solvers.
In the paper’s SFT setup:
- demonstrations are sourced from **easy difficulty** (hard serves as generalization)
- training uses **Qwen2.5-VL-7B-Instruct**, full-parameter finetuning, **global batch size 64**, **lr = 1e-5**, **bf16**
- **1500 steps** (single-task) / **5000 steps** (mixed-task)
- uses **LlamaFactory** for preprocessing + training orchestration

Please see **[visgym_training/README.md](visgym_training/README.md)** for more details regarding downloading the SFT data and training instructions.





# Our models

## (1) Released checkpoints
Download the ckpts here: https://huggingface.co/VisGym/visgym_model

You can use the hf CLI if you haven't installed it  
`curl -LsSf https://hf.co/cli/install.sh | bash`

then, for any model checkpoint you want to download:

`hf download VisGym/visgym_model --include <model_name>/*`

e.g., replacing `<model_name>` with `mixed_qwen3vl` if you want to evaluated our finetuned Qwen3-VL model under the mixed training setup.

## (2) Evaluate our model

After downloading the checkpoint(s), you can evaluate them using the provided script:

```bash
cd inference

# Basic usage (evaluates mixed_qwen3vl on easy split)
./eval_trained_model.sh

# Customize model, task, and split
MODEL=mixed_qwen3vl SPLIT=easy TASK=maze_3d ./eval_trained_model.sh

# Evaluate on hard split
MODEL=mixed_qwen3vl SPLIT=hard TASK=colorization ./eval_trained_model.sh

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 MODEL=mixed_qwen3vl ./eval_trained_model.sh
```

**Configuration Options:**
- `MODEL`: Model checkpoint name (default: `mixed_qwen3vl`)
- `SPLIT`: Difficulty split - `easy` or `hard` (default: `easy`)
- `TASK`: Task name without split suffix (default: `maze_3d`)
- `MODEL_ROOT`: Path to checkpoints directory (default: `../ckpts`)
- `DATASET_ROOT`: Path to inference dataset (default: `./inference_dataset`)
- `OUT_ROOT`: Output directory for results (default: `./eval_trained`)

The script will automatically:
- Validate all paths exist
- Set appropriate max steps based on split (20 for easy, 30 for hard)
- Generate GIFs and include environment feedback
- Save results to `inference/eval_trained/{model}_{task}_{split}/`




# Citation

If you find this repo useful, please cite:

```bibtex
@article{wang2026visgym,
  title        = {VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents},
  author       = {Wang, Zirui and Zhang, Junyi and Ge, Jiaxin and Lian, Long and Fu, Letian and Dunlap, Lisa and Goldberg, Ken and Wang, Xudong and Stoica, Ion and Chan, David M. and Min, Sewon and Gonzalez, Joseph E.},
  journal      = {arXiv preprint arXiv:2601.16973},
  year         = {2026},
  url          = {https://arxiv.org/abs/2601.16973}
}
```
