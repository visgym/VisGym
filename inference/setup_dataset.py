#!/usr/bin/env python3
"""
VLM-Gym Dataset Setup Script

Downloads the inference dataset from Hugging Face and sets it up in the
expected directory structure for running inference.

Usage:
    # Download everything to ./inference_dataset
    python setup_dataset.py

    # Download to a custom location
    python setup_dataset.py --output_dir /path/to/dataset

    # Download only easy difficulty (smaller, faster)
    python setup_dataset.py --difficulty easy

    # Download without large assets (for quick testing)
    python setup_dataset.py --no-assets

Example workflow:
    # 1. Download the dataset
    python setup_dataset.py --output_dir ./inference_dataset

    # 2. Run inference
    python run_inference.py \\
        --model "gpt-4o" \\
        --api_option openai \\
        --episodes_root ./inference_dataset/test_set_easy \\
        --initdir ./inference_dataset/initial_states_easy \\
        --outdir ./results
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


REPO_ID = "VisGym/inference-dataset"

# Define what each difficulty level needs
DIFFICULTY_PATTERNS = {
    "easy": [
        "test_set_easy/**",
        "initial_states_easy/**",
        "partial_datasets/**",  # Assets needed by some easy tasks
    ],
    "hard": [
        "test_set_hard/**",
        "initial_states_hard/**",
        "partial_datasets/**",
    ],
    "all": [
        "test_set_easy/**",
        "test_set_hard/**",
        "initial_states_easy/**",
        "initial_states_hard/**",
        "partial_datasets/**",
    ],
}

# Minimal download (no large assets)
MINIMAL_PATTERNS = {
    "easy": [
        "test_set_easy/**",
        "initial_states_easy/**",
    ],
    "hard": [
        "test_set_hard/**",
        "initial_states_hard/**",
    ],
    "all": [
        "test_set_easy/**",
        "test_set_hard/**",
        "initial_states_easy/**",
        "initial_states_hard/**",
    ],
}


def setup_dataset(
    output_dir: str = "./inference_dataset",
    difficulty: str = "all",
    include_assets: bool = True,
    token: str = None,
) -> Path:
    """
    Download and setup the VLM-Gym inference dataset.

    Args:
        output_dir: Where to download the dataset
        difficulty: "easy", "hard", or "all"
        include_assets: Whether to include large asset files (partial_datasets)
        token: HuggingFace token (optional, only needed for private repos)

    Returns:
        Path to the dataset directory
    """
    output_path = Path(output_dir).resolve()

    # Select patterns based on options
    if include_assets:
        patterns = DIFFICULTY_PATTERNS[difficulty]
    else:
        patterns = MINIMAL_PATTERNS[difficulty]

    print(f"=" * 60)
    print(f"VLM-Gym Dataset Setup")
    print(f"=" * 60)
    print(f"Repository: {REPO_ID}")
    print(f"Output dir: {output_path}")
    print(f"Difficulty: {difficulty}")
    print(f"Include assets: {include_assets}")
    print(f"Downloading: {', '.join(p.replace('/**', '') for p in patterns)}")
    print(f"=" * 60)
    print()

    # Download from HuggingFace
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(output_path),
        allow_patterns=patterns,
        token=token,
    )

    print()
    print("=" * 60)
    print(f"Download complete!")
    print(f"=" * 60)
    print()

    # Print what was downloaded
    print("Downloaded contents:")
    for item in sorted(output_path.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            # Count files in directory
            file_count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  {item.name}/ ({file_count} files)")
        else:
            print(f"  {item.name}")

    print()
    print_usage_instructions(output_path, difficulty)

    return output_path


def print_usage_instructions(dataset_path: Path, difficulty: str):
    """Print instructions on how to use the downloaded dataset."""

    if difficulty == "all":
        test_set = "test_set_easy"  # Default to easy for example
        init_dir = "initial_states_easy"
    else:
        test_set = f"test_set_{difficulty}"
        init_dir = f"initial_states_{difficulty}"

    print("=" * 60)
    print("Usage Instructions")
    print("=" * 60)
    print()
    print("Run inference with:")
    print()
    print(f"  python run_inference.py \\")
    print(f"      --model \"gpt-4o\" \\")
    print(f"      --api_option openai \\")
    print(f"      --episodes_root {dataset_path}/{test_set} \\")
    print(f"      --initdir {dataset_path}/{init_dir} \\")
    print(f"      --outdir ./results \\")
    print(f"      --render --make-gif")
    print()
    print("Or with OpenRouter:")
    print()
    print(f"  python run_inference.py \\")
    print(f"      --model \"anthropic/claude-sonnet-4\" \\")
    print(f"      --api_option openrouter \\")
    print(f"      --episodes_root {dataset_path}/{test_set} \\")
    print(f"      --initdir {dataset_path}/{init_dir} \\")
    print(f"      --outdir ./results")
    print()
    print("Environment variables needed:")
    print("  export OPENAI_API_KEY='sk-...'      # For OpenAI")
    print("  export OPENROUTER_API_KEY='or-...'  # For OpenRouter")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download and setup VLM-Gym inference dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything
  python setup_dataset.py

  # Download only easy difficulty
  python setup_dataset.py --difficulty easy

  # Quick download without large assets (for testing)
  python setup_dataset.py --no-assets

  # Custom output directory
  python setup_dataset.py --output_dir /data/vlm-gym
        """,
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./inference_dataset",
        help="Output directory (default: ./inference_dataset)",
    )

    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        choices=["easy", "hard", "all"],
        default="all",
        help="Which difficulty to download (default: all)",
    )

    parser.add_argument(
        "--no-assets",
        action="store_true",
        help="Skip downloading large asset files (partial_datasets). "
             "Note: Some tasks require these assets to run.",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (only needed for private repos)",
    )

    args = parser.parse_args()

    setup_dataset(
        output_dir=args.output_dir,
        difficulty=args.difficulty,
        include_assets=not args.no_assets,
        token=args.token,
    )


if __name__ == "__main__":
    main()
