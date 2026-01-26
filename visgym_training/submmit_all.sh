#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e


# Array of all script names to be executed
scripts=(
    # "colorization_100k.sh"
    # "counting-mixed_100k.sh"
    # "fetch_pick_place_100k.sh"
    # "fetch_reach_100k.sh"
    # "jigsaw-mixed_100k.sh"
    # "match_equation-mixed_100k.sh"
    # "matchstick_rotation_100k.sh"
    # "mental_rotation_2d_100k.sh"
    # "mental_rotation_3d_cube_100k.sh"
    # "mental_rotation_3d_objaverse_100k.sh"
    # "patch_reassembly_100k.sh"
    # "refdot_100k.sh"
    # "refdot_100k_cross.sh"
    # "sliding_block_100k.sh"
    # "toy_maze_2d_100k.sh"
    # "toy_maze_3d_100k.sh"
    # "video_unshuffle-mixed_100k.sh"
    # "zoom_in-mixed_100k.sh"
    # "counting/guess_only_100k.sh"
    # "counting/mark_all_100k.sh"
    # "jigsaw/reorder_100k.sh"
    # "jigsaw/swap_100k.sh"
    # "matchstick_equation/bfs_100k.sh"
    # "matchstick_equation/dfs_100k.sh"
    # "matchstick_equation/sos_100k.sh"
    # "video_unshuffle/reorder_100k.sh"
    # "video_unshuffle/swap_100k.sh"
    # "zoom_in_puzzle/reorder_100k.sh"
    # "zoom_in_puzzle/swap_100k.sh"
)

# Loop through each script in the array and launch it
for script_name in "${scripts[@]}"; do
    echo "=================================================="
    echo "==> Launching script: ${script_name}"
    echo "=================================================="
    python launch_local.py --run_script_path "./scripts/${script_name}"
done

echo "### All launch commands have been executed. ###"