#!/usr/bin/env python3
"""
Consolidate multiple conversation JSONL files into a single file.
Processes all conversation files in a directory and combines them.

Usage:
    python consolidate_conversations.py --input-dir /path/to/conversations --output-dir /path/to/output --remove_shards_after_consolidation
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any
import glob


def find_conversation_files(input_dir: str) -> List[str]:
    """
    Find all conversation JSONL files in a directory.
    
    Args:
        input_dir: Directory to search for conversation files
    
    Returns:
        List of paths to conversation JSONL files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    # Find all conversation JSONL files
    conversation_files = list(input_path.glob("*_conversations.jsonl"))
    
    if not conversation_files:
        print(f"Warning: No conversation JSONL files found in {input_dir}")
        return []
    
    # Sort files for consistent processing order
    conversation_files.sort()
    return [str(f) for f in conversation_files]


def consolidate_conversations(input_dir: str, output_dir: str, remove_shards: bool = False, shuffle: bool = True, random_seed: int = 42) -> Dict[str, Any]:
    """
    Consolidate all conversation files in a directory into a single file.
    
    Args:
        input_dir: Directory containing conversation JSONL files
        output_dir: Directory to save the consolidated file
        remove_shards: Whether to remove individual conversation files after consolidation
        shuffle: Whether to shuffle the conversations before writing
        random_seed: Random seed for shuffling
    
    Returns:
        Dictionary with consolidation results
    """
    # Find all conversation files
    conversation_files = find_conversation_files(input_dir)
    
    if not conversation_files:
        return {
            "processed_files": 0, 
            "total_conversations": 0, 
            "total_images": 0,
            "files_processed": [],
            "removed_files": []
        }
    
    print(f"Found {len(conversation_files)} conversation files to consolidate:")
    for file_path in conversation_files:
        print(f"  - {file_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    consolidated_file = output_path / "consolidated_conversations.jsonl"
    
    # Process all files
    results = {
        "processed_files": 0,
        "total_conversations": 0,
        "total_images": 0,
        "files_processed": [],
        "removed_files": [],
        "shuffled": shuffle
    }
    
    print(f"\nConsolidating conversations to: {consolidated_file}")
    if shuffle:
        print(f"Shuffling enabled with random seed: {random_seed}")
    
    # Collect all conversation data first
    all_conversations = []
    
    for file_path in conversation_files:
        try:
            print(f"Processing: {file_path}")
            
            # Read conversations from this file
            file_conversations = 0
            file_images = 0
            
            with open(file_path, 'r') as infile:
                for line_num, line in enumerate(infile):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse the conversation data
                        conversation_data = json.loads(line)
                        
                        # Add to collection
                        all_conversations.append(conversation_data)
                        
                        # Count conversations and images
                        file_conversations += len(conversation_data.get('conversations', []))
                        file_images += len(conversation_data.get('images', []))
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num + 1} in {file_path}: {e}")
                        continue
            
            results["files_processed"].append({
                "file": file_path,
                "conversations": file_conversations,
                "images": file_images
            })
            
            results["processed_files"] += 1
            results["total_conversations"] += file_conversations
            results["total_images"] += file_images
            
            print(f"  Conversations: {file_conversations}")
            print(f"  Images: {file_images}")
            
            # Remove the file if requested
            if remove_shards:
                try:
                    os.remove(file_path)
                    results["removed_files"].append(file_path)
                    print(f"  Removed: {file_path}")
                except Exception as e:
                    print(f"  Warning: Could not remove {file_path}: {e}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Shuffle conversations if requested
    if shuffle and all_conversations:
        print(f"Shuffling {len(all_conversations)} conversation entries...")
        random.seed(random_seed)
        random.shuffle(all_conversations)
    
    # Write all conversations to the consolidated file
    with open(consolidated_file, 'w') as outfile:
        for conversation_data in all_conversations:
            outfile.write(json.dumps(conversation_data) + '\n')
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate multiple conversation JSONL files into a single file"
    )
    
    parser.add_argument(
        "--input-dir", 
        required=True,
        help="Directory containing conversation JSONL files to consolidate"
    )
    
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Directory to save the consolidated file"
    )
    
    parser.add_argument(
        "--remove_shards_after_consolidation",
        action="store_true",
        help="Remove individual conversation files after consolidation"
    )
    
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of conversations (shuffling is enabled by default)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Consolidate conversations
    print(f"Consolidating conversation files from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Remove shards after consolidation: {args.remove_shards_after_consolidation}")
    print(f"Shuffle conversations: {not args.no_shuffle}")
    print(f"Random seed: {args.random_seed}")
    
    results = consolidate_conversations(
        args.input_dir, 
        args.output_dir, 
        args.remove_shards_after_consolidation,
        shuffle=not args.no_shuffle,
        random_seed=args.random_seed
    )
    
    print(f"\n=== CONSOLIDATION SUMMARY ===")
    print(f"Processed files: {results['processed_files']}")
    print(f"Total conversations: {results['total_conversations']}")
    print(f"Total images: {results['total_images']}")
    print(f"Shuffled: {results.get('shuffled', False)}")
    print(f"Consolidated file: {Path(args.output_dir) / 'consolidated_conversations.jsonl'}")
    
    if results.get('removed_files'):
        print(f"Removed files: {len(results['removed_files'])}")
        for file_path in results['removed_files']:
            print(f"  - {file_path}")
    
    if results.get('files_processed'):
        print(f"\nFile processing details:")
        for file_info in results['files_processed']:
            print(f"  {Path(file_info['file']).name}:")
            print(f"    Conversations: {file_info['conversations']}")
            print(f"    Images: {file_info['images']}")


if __name__ == "__main__":
    main()
