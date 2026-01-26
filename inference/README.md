# VLM-Gym Inference

Run VLM inference on pre-defined environment episodes to evaluate model performance.

## Download Dataset

The inference dataset is hosted on [Hugging Face](https://huggingface.co/datasets/VisGym/inference-dataset). Download it before running inference.

```bash
pip install huggingface_hub

# Download full dataset (~1.8GB)
python setup_dataset.py

# Or download only easy difficulty (~900MB)
python setup_dataset.py --difficulty easy

# Or download without large assets (~10MB, some tasks won't work)
python setup_dataset.py --no-assets
```

| Subset | Size | Command |
|--------|------|---------|
| Test sets only (no assets) | ~10 MB | `python setup_dataset.py --no-assets` |
| Easy (with assets) | ~900 MB | `python setup_dataset.py --difficulty easy` |
| Hard (with assets) | ~900 MB | `python setup_dataset.py --difficulty hard` |
| Full dataset | ~1.8 GB | `python setup_dataset.py` |

**Note:** Some tasks require assets from `partial_datasets/` (e.g., `counting`, `mental_rotation_3d_objaverse`). Use `--no-assets` only for tasks that don't need external data.

## Quick Start

```bash
# Set your API key
export OPENROUTER_API_KEY="your-api-key"
# or for OpenAI models
export OPENAI_API_KEY="your-api-key"

# Run inference
python run_inference.py \
    --model "qwen/qwen3-vl-235b-a22b-instruct" \
    --api_option openrouter \
    --episodes_root ./inference_dataset/test_set_easy \
    --initdir ./inference_dataset/initial_states_easy \
    --outdir ./eval/qwen3_easy \
    --make-gif
```

## Using the Experiment Script

The `run_experiments.sh` script provides a convenient way to run experiments:

```bash
# Run default model (qwen3) on all splits
./run_experiments.sh

# Run specific model
MODELS="claude" ./run_experiments.sh

# Run multiple models
MODELS="qwen3 gemini claude" ./run_experiments.sh

# Run only easy split
SPLITS="easy" ./run_experiments.sh

# Run specific model on specific split
MODELS="gpt5" SPLITS="hard" ./run_experiments.sh
```

## Supported Models

| Tag | Model Name | API |
|-----|------------|-----|
| `gpt5` | gpt-5 | openai |
| `gemini` | google/gemini-2.5-pro | openrouter |
| `qwen3` | qwen/qwen3-vl-235b-a22b-instruct | openrouter |
| `qwen` | qwen/qwen2.5-vl-72b-instruct | openrouter |
| `qwenvlmax` | qwen/qwen-vl-max | openrouter |
| `claude` | anthropic/claude-sonnet-4 | openrouter |
| `grok` | x-ai/grok-4-fast | openrouter |
| `glm` | z-ai/glm-4.5v | openrouter |
| `intern` | opengvlab/internvl3-78b | openrouter |
| `llama` | meta-llama/llama-4-maverick | openrouter |
| `gemma` | google/gemma-3-27b-it | openrouter |
| `mistral` | mistralai/mistral-medium-3.1 | openrouter |
| `uitars` | bytedance/ui-tars-1.5-7b | openrouter |

## Dataset Structure

```
inference_dataset/
├── test_set_easy/           # Easy split episodes (JSONL files)
│   ├── colorization__easy/
│   ├── counting__easy/
│   ├── fetch_reach__easy/
│   ├── jigsaw__easy/
│   ├── matchstick_equation__easy/
│   ├── matchstick_rotation__easy/
│   ├── maze_2d__easy/
│   ├── maze_3d__easy/
│   ├── mental_rotation_2d__easy/
│   ├── mental_rotation_3d_cube__easy/
│   ├── mental_rotation_3d_objaverse__easy/
│   ├── patch_reassembly__easy/
│   ├── referring_dot_pointing__easy/
│   ├── sliding_block__easy/
│   ├── video_unshuffle__easy/
│   └── zoom_in_puzzle__easy/
├── test_set_hard/           # Hard split episodes
│   └── ...
├── initial_states_easy/     # Initial states for easy split
│   └── ...
├── initial_states_hard/     # Initial states for hard split
│   └── ...
└── partial_datasets/        # Supporting data (images, etc.)
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name (required) | - |
| `--api_option` | API backend: `openai`, `openrouter`, `transformers` | `openrouter` |
| `--episodes_root` | Path to JSONL episode files (required) | - |
| `--initdir` | Directory containing initial states | `./inference_dataset/initial_states` |
| `--outdir` | Output directory | `runs/infer` |
| `--max-steps` | Maximum steps per episode | `1` |
| `--render` | Save per-step render images | `false` |
| `--make-gif` | Generate GIF per episode | `false` |
| `--include-env-feedback` | Include textual environment feedback | `false` |
| `--elicit-verbalization` | Ask model for reasoning verbalization | `false` |
| `--num-interaction-history` | History steps to include (-1 = all) | `-1` |
| `--verbose` | Enable verbose output | `false` |
| `--save-verbose` | Save step-by-step history.jsonl | `false` |
| `--env-filter` | Only run specific env IDs | `null` |
| `--use-3d-maze` | Run only 3D maze environments | `false` |

## Output Structure

```
eval/
└── qwen3_easy/
    └── jigsaw__easy/
        ├── seed1687216_episode309018093/
        │   ├── episode_stats.json    # Results and metadata
        │   ├── history.jsonl         # Step-by-step history (if --save-verbose)
        │   └── renders/              # Render images (if --render)
        └── seed1687216_episode309018093_r1.000.gif  # Episode GIF (if --make-gif)
```

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI models
- `OPENROUTER_API_KEY`: Required for OpenRouter models

