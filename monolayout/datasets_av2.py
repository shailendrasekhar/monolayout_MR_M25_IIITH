"""
Argoverse 2 Dataset Classes for MonoLayout

This module provides dataset classes specifically designed for Argoverse 2 data format.
Key differences from Argoverse 1:
- Different directory structure (sensors/cameras/, sensors/lidar/)
- Different file naming conventions
- Uses .feather files for annotations
- Different camera naming (ring_front_center vs stereo_front_left)
"""

from __future__ import absolute_import, division, print_function

import os
import random
from pathlib import Path

import PIL.Image as pil
import numpy as np
import torch.utils.data as data
from torchvision import transforms

# Import base classes from original datasets
from .datasets import MonoDataset, process_topview, process_discr, resize_topview


class ArgoverseV2(MonoDataset):
    """Argoverse 2 dataset class for MonoLayout."""
    
    def __init__(self, *args, **kwargs):
        super(ArgoverseV2, self).__init__(*args, **kwargs)
        self.root_dir = "./data/argoverse2"
        
        # Argoverse 2 specific camera name
        self.camera_name = "ring_front_center"
        
        # Map old camera name to new one for backward compatibility
        if hasattr(self.opt, 'camera_name'):
            self.camera_name = self.opt.camera_name
    
    def get_image_path(self, root_dir, frame_index):
        """
        Get image path for Argoverse 2 format.
        
        Args:
            root_dir: Root directory path
            frame_index: Frame identifier in format "log_id/timestamp_ns"
        
        Returns:
            Path to the image file
        """
        # Parse frame_index to extract log_id and timestamp
        if isinstance(frame_index, str) and '/' in frame_index:
            log_id, timestamp_ns = frame_index.split('/', 1)
        else:
            # Handle legacy format - try to extract from road_gt path
            if "road_gt" in str(frame_index):
                # Convert from road_gt format to image format
                parts = str(frame_index).split('/')
                log_id = parts[-2]  # Second to last part should be log_id
                filename = parts[-1]  # Last part is filename
                # Extract timestamp from filename
                if self.camera_name in filename:
                    timestamp_ns = filename.replace(f"{self.camera_name}_", "").replace(".png", "").replace(".jpg", "")
                else:
                    # Fallback: assume filename is timestamp
                    timestamp_ns = filename.replace(".png", "").replace(".jpg", "")
            else:
                raise ValueError(f"Cannot parse frame_index: {frame_index}")
        
        # Construct image path
        img_path = os.path.join(
            root_dir, 
            log_id, 
            "sensors", 
            "cameras", 
            self.camera_name, 
            f"{timestamp_ns}.jpg"
        )
        
        return img_path
    
    def get_static_path(self, root_dir, frame_index):
        """
        Get static (road) layout path for training.
        
        For Argoverse 2, this would be weak supervision generated from map data.
        """
        # Parse frame_index
        if isinstance(frame_index, str) and '/' in frame_index:
            log_id, timestamp_ns = frame_index.split('/', 1)
        else:
            # Handle string format like "log_id/road_bev/camera_timestamp.png"
            parts = str(frame_index).split('/')
            log_id = parts[0] if len(parts) > 1 else frame_index
            if len(parts) > 2:
                filename = parts[-1]
                timestamp_ns = filename.replace(f"{self.camera_name}_", "").replace(".png", "")
            else:
                timestamp_ns = frame_index
        
        path = os.path.join(
            root_dir, 
            log_id, 
            "road_bev", 
            f"{self.camera_name}_{timestamp_ns}.png"
        )
        return path
    
    def get_dynamic_path(self, root_dir, frame_index):
        """
        Get dynamic (vehicle) layout path for training.
        """
        # Parse frame_index similar to static path
        if isinstance(frame_index, str) and '/' in frame_index:
            log_id, timestamp_ns = frame_index.split('/', 1)
        else:
            parts = str(frame_index).split('/')
            log_id = parts[0] if len(parts) > 1 else frame_index
            if len(parts) > 2:
                filename = parts[-1]
                timestamp_ns = filename.replace(f"{self.camera_name}_", "").replace(".jpg", "")
            else:
                timestamp_ns = frame_index
        
        path = os.path.join(
            root_dir, 
            log_id, 
            "car_bev_gt", 
            f"{self.camera_name}_{timestamp_ns}.jpg"
        )
        return path
    
    def get_static_gt_path(self, root_dir, frame_index):
        """
        Get static (road) ground truth path for evaluation.
        """
        # Parse frame_index
        if isinstance(frame_index, str) and '/' in frame_index:
            log_id, timestamp_ns = frame_index.split('/', 1)
        else:
            parts = str(frame_index).split('/')
            log_id = parts[0] if len(parts) > 1 else frame_index
            if len(parts) > 2:
                filename = parts[-1]
                timestamp_ns = filename.replace(f"{self.camera_name}_", "").replace(".png", "")
            else:
                timestamp_ns = frame_index
        
        path = os.path.join(
            root_dir, 
            log_id, 
            "road_gt", 
            f"{self.camera_name}_{timestamp_ns}.png"
        )
        return path
    
    def get_dynamic_gt_path(self, root_dir, frame_index):
        """
        Get dynamic (vehicle) ground truth path for evaluation.
        """
        return self.get_dynamic_path(root_dir, frame_index)
    
    def get_osm_path(self, root_dir):
        """
        Get OpenStreetMap path for discriminator training.
        
        For now, we'll use the same OSM data as the original implementation.
        """
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)
        return osm_path


def create_argoverse2_filelist(data_path: str, split: str = "train", 
                              max_logs: int = None, max_frames_per_log: int = None):
    """
    Create file list for Argoverse 2 dataset.
    
    Args:
        data_path: Path to Argoverse 2 dataset
        split: Dataset split ('train', 'val', 'test')
        max_logs: Maximum number of logs to include (for testing)
        max_frames_per_log: Maximum frames per log (for testing)
    
    Returns:
        List of frame identifiers in format "log_id/timestamp_ns"
    """
    split_path = os.path.join(data_path, split)
    
    if not os.path.exists(split_path):
        raise ValueError(f"Split path does not exist: {split_path}")
    
    filelist = []
    log_dirs = [d for d in os.listdir(split_path) 
                if os.path.isdir(os.path.join(split_path, d)) and not d.startswith('.')]
    
    if max_logs:
        log_dirs = log_dirs[:max_logs]
    
    print(f"Processing {len(log_dirs)} logs from {split} split")
    
    for log_id in log_dirs:
        log_path = os.path.join(split_path, log_id)
        
        # Check if this log has the required camera
        camera_path = os.path.join(log_path, "sensors", "cameras", "ring_front_center")
        
        if not os.path.exists(camera_path):
            print(f"Warning: Camera path not found for log {log_id}")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(camera_path) if f.endswith('.jpg')]
        
        if max_frames_per_log:
            image_files = image_files[:max_frames_per_log]
        
        # Add to filelist
        for img_file in image_files:
            timestamp_ns = img_file.replace('.jpg', '')
            frame_id = f"{log_id}/{timestamp_ns}"
            filelist.append(frame_id)
    
    print(f"Created filelist with {len(filelist)} frames")
    return filelist


def get_argoverse2_splits(data_path: str, split: str = "train"):
    """
    Get train/val splits for Argoverse 2.
    
    Args:
        data_path: Path to Argoverse 2 dataset
        split: Which split to return
    
    Returns:
        List of frame identifiers
    """
    return create_argoverse2_filelist(data_path, split)


# Convenience function to create dataset
def create_argoverse2_dataset(opt, split="train", max_logs=None):
    """
    Create Argoverse 2 dataset instance.
    
    Args:
        opt: Options object with dataset parameters
        split: Dataset split
        max_logs: Maximum number of logs (for testing)
    
    Returns:
        ArgoverseV2 dataset instance
    """
    # Create filelist
    filenames = create_argoverse2_filelist(
        opt.data_path, 
        split=split, 
        max_logs=max_logs
    )
    
    # Create dataset
    is_train = (split == "train")
    dataset = ArgoverseV2(opt, filenames, is_train=is_train)
    
    return dataset
