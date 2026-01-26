#!/bin/bash

# Script to download data from Hugging Face
# Usage: bash data_transfer.sh [dataset_id] [location] [filter_script]

DATASET_REPO="https://huggingface.co/datasets/visgym/visgym_data"
TARGET_DIR="/home/clouduser/Code/data/gym" # Adjust if running locally without docker mapping to this path

# If running locally (not in the specific docker path structure), maybe we should default to ./data?
# But the scripts seem to hardcode /home/clouduser/Code/data/gym...
# Let's try to detect if we are in the docker container or not, or just use a relative path if the absolute one doesn't exist?
# For now, I'll stick to the requested URL and a sensible default, but I'll add a check.

if [ ! -d "/home/clouduser/Code/data/gym" ]; then
    # Fallback for local execution if the specific path doesn't exist
    TARGET_DIR="./data/gym"
fi

mkdir -p "$TARGET_DIR"

echo "Downloading data from $DATASET_REPO to $TARGET_DIR..."

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs is not installed. Please install it first."
    # Try to install if we can (e.g. in docker)
    if [ -f /etc/debian_version ]; then
        apt-get update && apt-get install -y git-lfs
        git lfs install
    fi
fi

# Clone the repo
# We use a temporary directory to clone the repo, then move the files to the target structure
TEMP_DIR=$(mktemp -d)
git clone "$DATASET_REPO" "$TEMP_DIR"

# Move contents to target dir
# The structure of the repo is assumed to match what the scripts expect
# If the repo contains the 'gym' folder content directly:
cp -r "$TEMP_DIR"/* "$TARGET_DIR/"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Data download complete."
