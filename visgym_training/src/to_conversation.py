#!/usr/bin/env python3
"""
Convert saved episode JSONL files to conversation format.
Processes JSONL files in a directory with multithreading support.

Usage:
    python convert_episode_to_conversation.py \
        --input-dir /path/to/jsonl/files \
        --output-dir /path/to/output \
        --dataset-name my_dataset \
        --num-trajectories 1000 \
        --num-workers 4 \
        [--remove-zero-reward] \
        [--remove-inference-contamination]
"""

import argparse
import json
import base64
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
import glob
import concurrent.futures
import multiprocessing


def extract_base64_from_data_url(data_url: str) -> str:
    """Extract base64 data from a data URL."""
    if data_url.startswith('data:image/'):
        # Extract the base64 part after the comma
        return data_url.split(',', 1)[1]
    return data_url


def save_base64_image(base64_data: str, output_dir: Path, episode_num: int, step_num: int) -> str:
    """Save base64 image data to a JPG file."""
    try:
        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        
        # Save as JPG with episode and step info
        image_name = f"episode_{episode_num:03d}_step_{step_num:03d}"
        image_path = output_dir / f"{image_name}.jpg"
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        return str(image_path)
    except Exception as e:
        print(f"Warning: Could not save image episode_{episode_num}_step_{step_num}: {e}")
        return ""


def should_filter_episode(episode_data: Dict[str, Any], remove_zero_reward: bool, forbidden_hashes: "set[str]" = None) -> bool:
    """
    Determine if an episode should be filtered out.
    
    Args:
        episode_data: Dictionary containing episode data
        remove_zero_reward: Whether to remove samples with reward != 1.0
        forbidden_hashes: Set of hashes to exclude (inference contamination)
    
    Returns:
        bool: True if episode should be filtered out, False otherwise
    """
    # Check zero reward filter
    if remove_zero_reward:
        stats = episode_data.get('stats', {})
        reward = stats.get('reward')
        if reward != 1.0:
            return True
    
    # Check inference contamination via hash
    if forbidden_hashes is not None and len(forbidden_hashes) > 0:
        sample_hash = episode_data.get('hash')
        if sample_hash in forbidden_hashes:
            return True
    
    return False


def convert_episode_to_conversation(episode_data: Dict[str, Any], output_dir: Path, episode_num: int) -> Dict[str, Any]:
    """
    Convert a single episode data to conversation format.
    
    Args:
        episode_data: Dictionary containing episode data
        output_dir: Directory to save images
        episode_num: Episode number for naming
    
    Returns:
        Dictionary in the conversation format
    """
    # Extract history
    history = episode_data.get('history', [])
    
    # Initialize conversation
    conversations = []
    images = []
    
    for i, step in enumerate(history):
        # Get the prompt and VLM output
        prompt = step.get('prompt', '')
        vlm_output = step.get('vlm_output', '')
        image_b64 = step.get('image', '')
        
        # Skip if no image
        if not image_b64:
            continue
        
        # Extract base64 data
        base64_data = extract_base64_from_data_url(image_b64)
        
        # Save image
        image_path = save_base64_image(base64_data, output_dir, episode_num, i)
        
        if image_path:
            # Add human message (prompt with image)
            conversations.append({
                "from": "human",
                "value": f"<image>{prompt}"
            })
            
            # Add assistant message (VLM output)
            conversations.append({
                "from": "gpt", 
                "value": vlm_output
            })
            
            # Add image path
            images.append(image_path)
    
    # Create the final format
    result = {
        "episode": episode_num,
        "timestamp": episode_data.get('timestamp', ''),
        "conversations": conversations,
        "images": images,
        "stats": episode_data.get('stats', {})
    }
    
    return result


def find_jsonl_files(input_dir: str) -> List[str]:
    """
    Find all JSONL files in a directory.
    
    Args:
        input_dir: Directory to search for JSONL files
    
    Returns:
        List of paths to JSONL files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    # Find all .jsonl files in the directory
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"Warning: No JSONL files found in {input_dir}")
        return []
    
    # Sort files for consistent processing order
    jsonl_files.sort()
    return [str(f) for f in jsonl_files]


def convert_episodes_jsonl_to_conversations(episodes_path: str, output_dir: Path, 
                                           remove_zero_reward: bool = False,
                                           max_trajectories: int = None,
                                           forbidden_hashes: "set[str]" = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Convert a JSONL file containing multiple episodes to conversation format.
    
    Args:
        episodes_path: Path to the episodes JSONL file
        output_dir: Directory to save images
        remove_zero_reward: Whether to remove samples with reward != 1.0
        max_trajectories: Maximum number of trajectories to process (None = all)
    
    Returns:
        Tuple of (list of conversation dicts, stats counters dict)
    """
    conversations = []
    total, reward_fltered, forbidden_fltered, remained = 0, 0, 0, 0
    
    # Read the JSONL file line by line
    with open(episodes_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the episode data
                episode_data = json.loads(line)
                total += 1
                
                # Reward-based filtering (count separately)
                if remove_zero_reward:
                    stats = episode_data.get('stats', {})
                    reward = stats.get('reward')
                    if reward != 1.0:
                        reward_fltered += 1
                        continue
                
                # Inference contamination filtering via hash (count separately)
                if forbidden_hashes:
                    sample_hash = episode_data.get('hash')
                    if sample_hash in forbidden_hashes:
                        forbidden_fltered += 1
                        continue
                
                # Convert to conversation format
                conversation = convert_episode_to_conversation(
                    episode_data, output_dir, episode_data.get('episode', line_num)
                )
                
                conversations.append(conversation)
                remained += 1
                
                # Check if we've reached the maximum number of trajectories
                if max_trajectories and len(conversations) >= max_trajectories:
                    break
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num + 1}: {e}")
                continue
    
    return conversations, {
        "total": total,
        "reward_filtered": reward_fltered,
        "forbidden_filtered": forbidden_fltered,
        "remained": remained,
    }


def process_single_file(file_path: str, output_base_dir: Path, dataset_name: str, batch_num: int,
                       remove_zero_reward: bool = False, max_trajectories: int = None,
                       forbidden_hashes: "set[str]" = None) -> Dict[str, Any]:
    """
    Process a single JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        output_base_dir: Base directory for output
        dataset_name: Name of the dataset
        batch_num: Batch number for naming
        remove_zero_reward: Whether to remove samples with reward != 1.0
        max_trajectories: Maximum number of trajectories to process (None = all)
    
    Returns:
        Dictionary with processing results for this file
    """
    try:
        # Create output directory for images
        file_output_dir = output_base_dir / f"{dataset_name}_batch_{batch_num:06d}"
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {file_path}")
        print(f"Images directory: {file_output_dir}")
        
        # Convert the file
        conversations, counters = convert_episodes_jsonl_to_conversations(
            file_path, file_output_dir, remove_zero_reward, max_trajectories, forbidden_hashes
        )
        
        # Save conversations as JSONL
        output_file = output_base_dir / f"{dataset_name}_batch_{batch_num:06d}_conversations.jsonl"
        with open(output_file, 'w') as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + '\n')
        
        # Calculate stats for this file
        file_conversations = sum(len(conv['conversations']) for conv in conversations)
        file_images = sum(len(conv['images']) for conv in conversations)
        
        file_result = {
            "input_file": file_path,
            "output_file": str(output_file),
            "output_dir": str(file_output_dir),
            "episodes": len(conversations),
            "conversations": file_conversations,
            "images": file_images,
            "total": counters.get("total", 0),
            "reward_filtered": counters.get("reward_filtered", 0),
            "forbidden_filtered": counters.get("forbidden_filtered", 0),
            "remained": counters.get("remained", 0),
        }
        
        print(f"  Episodes: {len(conversations)}")
        print(f"  Conversations: {file_conversations}")
        print(f"  Images: {file_images}")
        print(f"  Total read: {file_result['total']} | Reward filtered: {file_result['reward_filtered']} | Forbidden filtered: {file_result['forbidden_filtered']} | Remained: {file_result['remained']}")
        print(f"  Conversations saved to: {output_file}")
        print(f"  Images saved to: {file_output_dir}")
        
        return file_result
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_directory(input_dir: str, output_base_dir: str, dataset_name: str, 
                     num_trajectories: int, trajectories_per_batch: int, 
                     num_workers: int, remove_zero_reward: bool = False,
                     forbidden_hashes: "set[str]" = None) -> Dict[str, Any]:
    """
    Process JSONL files in a directory with multithreading.
    
    Args:
        input_dir: Directory containing JSONL files
        output_base_dir: Base directory for output
        dataset_name: Name of the dataset
        num_trajectories: Total number of trajectories to process
        trajectories_per_batch: Number of trajectories per batch
        num_workers: Number of worker threads (0 = all available)
        remove_zero_reward: Whether to remove samples with reward != 1.0
    
    Returns:
        Dictionary with processing results
    """
    # Find all JSONL files and sort by batch number
    jsonl_files = find_jsonl_files(input_dir)
    
    if not jsonl_files:
        return {"processed_files": 0, "total_conversations": 0, "total_images": 0}
    
    # Sort files by batch number (extract number from filename)
    def extract_batch_number(file_path):
        filename = Path(file_path).stem
        match = re.search(r'batch_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    jsonl_files.sort(key=extract_batch_number)
    
    # Calculate how many files to process
    files_to_process = min(len(jsonl_files), (num_trajectories + trajectories_per_batch - 1) // trajectories_per_batch)
    selected_files = jsonl_files[:files_to_process]
    
    print(f"Found {len(jsonl_files)} JSONL files, processing {files_to_process} files:")
    for i, file_path in enumerate(selected_files):
        print(f"  - {file_path} (batch {i:06d})")
    
    # Create output directory
    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if num_workers == 0:
        num_workers = multiprocessing.cpu_count()
    
    print(f"Remove zero reward samples: {remove_zero_reward}")
    if forbidden_hashes is not None:
        print(f"Remove inference contamination: True (forbidden hashes: {len(forbidden_hashes)})")
    else:
        print(f"Remove inference contamination: False")
    print(f"Using {num_workers} workers for processing")
    
    # Process files with multithreading
    results = {
        "processed_files": 0,
        "total_conversations": 0,
        "total_images": 0,
        "total_read": 0,
        "total_reward_filtered": 0,
        "total_forbidden_filtered": 0,
        "total_remained": 0,
        "files": []
    }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {}
        for i, file_path in enumerate(selected_files):
            future = executor.submit(process_single_file, file_path, output_path, dataset_name, i, remove_zero_reward, None, forbidden_hashes)
            future_to_file[future] = file_path
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_result = future.result()
                if file_result:
                    results["files"].append(file_result)
                    results["processed_files"] += 1
                    results["total_conversations"] += file_result["conversations"]
                    results["total_images"] += file_result["images"]
                    results["total_read"] += file_result.get("total", 0)
                    results["total_reward_filtered"] += file_result.get("reward_filtered", 0)
                    results["total_forbidden_filtered"] += file_result.get("forbidden_filtered", 0)
                    results["total_remained"] += file_result.get("remained", 0)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert saved episodes JSONL to conversation format with multithreading"
    )
    
    parser.add_argument(
        "--input-dir", 
        required=True,
        help="Directory containing JSONL files to process"
    )
    
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Output directory for conversations and images"
    )
    
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name of the dataset (used as prefix for batch files)"
    )
    
    parser.add_argument(
        "--num-trajectories",
        type=int,
        required=True,
        help="Total number of trajectories to process"
    )
    
    parser.add_argument(
        "--trajectories-per-batch",
        type=int,
        default=100,
        help="Number of trajectories per batch (default: 100)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="Number of worker threads (0 = use all available CPU cores)"
    )
    
    parser.add_argument(
        "--remove-zero-reward",
        action="store_true",
        help="Remove samples with reward != 1.0"
    )
    
    parser.add_argument(
        "--remove-inference-contamination",
        action="store_true",
        help="Remove samples whose hash appears in the provided merged test-set hash file"
    )
    
    parser.add_argument(
        "--inference-hash-file",
        required=False,
        help="Absolute path to the merged inference hash JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_trajectories <= 0:
        parser.error("--num-trajectories must be positive")
    
    if args.trajectories_per_batch <= 0:
        parser.error("--trajectories-per-batch must be positive")
    
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    
    # Build forbidden hashes set if requested
    forbidden_hashes = None
    if args.remove_inference_contamination:
        if not args.inference_hash_file:
            parser.error("--inference-hash-file is required when --remove-inference-contamination is set")

        def _load_hash_file(abs_path: str) -> Dict[str, List[str]]:
            with open(abs_path, 'r') as jf:
                return json.load(jf)

        forbidden_hashes = set()
        try:
            merged_map = _load_hash_file(args.inference_hash_file)
            for values in merged_map.values():
                if isinstance(values, list):
                    forbidden_hashes.update(values)
        except Exception as e:
            print(f"Warning: failed loading hashes from {args.inference_hash_file}: {e}")

        print(f"Loaded {len(forbidden_hashes)} forbidden hashes from {args.inference_hash_file}")

    # Process directory
    print(f"Processing JSONL files in directory: {args.input_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Number of trajectories: {args.num_trajectories}")
    print(f"Trajectories per batch: {args.trajectories_per_batch}")
    print(f"Number of workers: {args.num_workers if args.num_workers > 0 else 'all available'}")
    
    results = process_directory(
        args.input_dir, 
        args.output_dir, 
        args.dataset_name,
        args.num_trajectories,
        args.trajectories_per_batch,
        args.num_workers,
        args.remove_zero_reward,
        forbidden_hashes
    )
    
    print(f"\n=== SUMMARY ===")
    print(f"Processed files: {results['processed_files']}")
    print(f"Total conversations: {results['total_conversations']}")
    print(f"Total images: {results['total_images']}")
    if 'total_read' in results:
        print(f"Total read: {results['total_read']} | Reward filtered: {results['total_reward_filtered']} | Forbidden filtered: {results['total_forbidden_filtered']} | Remained: {results['total_remained']}")
    
    if results['files']:
        print(f"\nFile details:")
        for file_result in results['files']:
            print(f"  {Path(file_result['input_file']).name}:")
            print(f"    Episodes: {file_result['episodes']}")
            print(f"    Conversations: {file_result['conversations']}")
            print(f"    Images: {file_result['images']}")
            print(f"    Output: {file_result['output_file']}")
            if 'total' in file_result:
                print(f"    Total read: {file_result['total']} | Reward filtered: {file_result['reward_filtered']} | Forbidden filtered: {file_result['forbidden_filtered']} | Remained: {file_result['remained']}")


if __name__ == "__main__":
    main() 