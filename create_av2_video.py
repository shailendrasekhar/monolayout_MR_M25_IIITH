#!/usr/bin/env python3
"""
Create video from Argoverse 2 BEV visualizations
Shows time-lapse of camera + BEV ground truth
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse


def create_video_from_visualizations(vis_dir, output_video, fps=10, max_frames=None):
    """
    Create video from Argoverse 2 visualization images
    
    Args:
        vis_dir: Directory containing visualization PNG files
        output_video: Output video file path
        fps: Frames per second
        max_frames: Maximum number of frames to include
    """
    vis_dir = Path(vis_dir)
    
    # Get sorted list of visualization files
    vis_files = sorted(list(vis_dir.glob('vis_*.png')))
    
    if max_frames:
        vis_files = vis_files[:max_frames]
    
    if not vis_files:
        print(f"No visualization files found in {vis_dir}!")
        return False
    
    print(f"Creating video from {len(vis_files)} frames at {fps} FPS")
    print(f"Input directory: {vis_dir}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(vis_files[0]))
    if first_frame is None:
        print(f"Error reading first frame: {vis_files[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    print(f"Frame dimensions: {width}x{height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open video writer")
        return False
    
    frame_count = 0
    for vis_file in vis_files:
        # Read visualization image
        frame = cv2.imread(str(vis_file))
        
        if frame is None:
            print(f"Warning: Could not read {vis_file}")
            continue
        
        # Resize if needed
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # Add frame counter and filename
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Frame {frame_count + 1}/{len(vis_files)}', 
                   (10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Add timestamp from filename
        timestamp = vis_file.stem.split('_')[-1]
        cv2.putText(frame, f'Timestamp: {timestamp}', 
                   (10, height - 10), font, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{len(vis_files)} frames")
    
    out.release()
    print(f"\n✅ Video saved: {output_video}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count / fps:.1f} seconds")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Create video from Argoverse 2 BEV visualizations')
    parser.add_argument('--vis_dir', type=str, required=True,
                        help='Directory containing visualization PNG files')
    parser.add_argument('--output', type=str, default='argoverse2_bev.mp4',
                        help='Output video file')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second (default: 10)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames')
    
    args = parser.parse_args()
    
    success = create_video_from_visualizations(
        args.vis_dir, 
        args.output, 
        args.fps, 
        args.max_frames
    )
    
    if not success:
        print("❌ Video creation failed")
        exit(1)


if __name__ == '__main__':
    main()
