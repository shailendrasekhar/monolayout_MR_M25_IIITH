#!/usr/bin/env python3
"""
Create train/validation splits for Argoverse 2 dataset.

This script generates the file lists needed for MonoLayout training with Argoverse 2.
"""

import os
import argparse
from pathlib import Path


def create_argo2_splits(data_path: str, output_dir: str = "splits/argo2", 
                       val_ratio: float = 0.1, max_logs: int = None):
    """
    Create train/val splits for Argoverse 2.
    
    Args:
        data_path: Path to Argoverse 2 dataset
        output_dir: Output directory for split files
        val_ratio: Ratio of data to use for validation
        max_logs: Maximum number of logs to process (for testing)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get train split path
    train_split_path = os.path.join(data_path, "train")
    
    if not os.path.exists(train_split_path):
        raise ValueError(f"Train split path does not exist: {train_split_path}")
    
    # Get all log directories
    log_dirs = [d for d in os.listdir(train_split_path) 
                if os.path.isdir(os.path.join(train_split_path, d)) and not d.startswith('.')]
    
    if max_logs:
        log_dirs = log_dirs[:max_logs]
        print(f"Limited to {max_logs} logs for testing")
    
    print(f"Found {len(log_dirs)} logs in train split")
    
    # Collect all frame identifiers
    all_frames = []
    
    for log_id in log_dirs:
        log_path = os.path.join(train_split_path, log_id)
        
        # Check if this log has the required camera
        camera_path = os.path.join(log_path, "sensors", "cameras", "ring_front_center")
        
        if not os.path.exists(camera_path):
            print(f"Warning: Camera path not found for log {log_id}, skipping")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(camera_path) if f.endswith('.jpg')]
        
        # Add to frame list
        for img_file in image_files:
            timestamp_ns = img_file.replace('.jpg', '')
            frame_id = f"{log_id}/{timestamp_ns}"
            all_frames.append(frame_id)
    
    print(f"Total frames collected: {len(all_frames)}")
    
    # Split into train and validation
    num_val = int(len(all_frames) * val_ratio)
    
    # Shuffle for random split
    import random
    random.seed(42)  # For reproducible splits
    random.shuffle(all_frames)
    
    val_frames = all_frames[:num_val]
    train_frames = all_frames[num_val:]
    
    print(f"Train frames: {len(train_frames)}")
    print(f"Val frames: {len(val_frames)}")
    
    # Write train split
    train_file = os.path.join(output_dir, "train_files.txt")
    with open(train_file, 'w') as f:
        for frame in train_frames:
            f.write(f"{frame}\n")
    
    # Write val split
    val_file = os.path.join(output_dir, "val_files.txt")
    with open(val_file, 'w') as f:
        for frame in val_frames:
            f.write(f"{frame}\n")
    
    print(f"Split files created:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    
    return train_frames, val_frames


def main():
    parser = argparse.ArgumentParser(description="Create Argoverse 2 splits for MonoLayout")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to Argoverse 2 dataset root")
    parser.add_argument("--output_dir", type=str, default="splits/argo2",
                        help="Output directory for split files")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio of data to use for validation")
    parser.add_argument("--max_logs", type=int, default=None,
                        help="Maximum number of logs to process (for testing)")
    
    args = parser.parse_args()
    
    create_argo2_splits(
        data_path=args.data_path,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        max_logs=args.max_logs
    )


if __name__ == "__main__":
    main()
