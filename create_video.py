#!/usr/bin/env python3
"""
Create video/GIF from inference results
Shows time-lapse of predictions
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse


def create_video_from_predictions(input_dir, bev_dir, output_video, fps=10, max_frames=None):
    """
    Create video showing input images and BEV predictions side-by-side
    """
    input_dir = Path(input_dir)
    bev_dir = Path(bev_dir)
    
    # Get sorted list of BEV predictions
    bev_files = sorted(list(bev_dir.glob('*.png')))
    
    if max_frames:
        bev_files = bev_files[:max_frames]
    
    if not bev_files:
        print("No BEV files found!")
        return False
    
    print(f"Creating video from {len(bev_files)} frames at {fps} FPS")
    
    # Read first frame to get dimensions
    first_bev = cv2.imread(str(bev_files[0]), cv2.IMREAD_GRAYSCALE)
    target_height = 480
    target_width = 960
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (target_width * 2, target_height))
    
    frame_count = 0
    for bev_file in bev_files:
        # Find corresponding input
        input_file = input_dir / (bev_file.stem + '.jpg')
        if not input_file.exists():
            input_file = input_dir / (bev_file.stem + '.png')
        
        if not input_file.exists():
            continue
        
        # Read images
        input_img = cv2.imread(str(input_file))
        bev_img = cv2.imread(str(bev_file), cv2.IMREAD_GRAYSCALE)
        
        if input_img is None or bev_img is None:
            continue
        
        # Resize
        input_resized = cv2.resize(input_img, (target_width, target_height))
        
        # Color BEV
        bev_colored = cv2.applyColorMap(bev_img, cv2.COLORMAP_SPRING)
        bev_resized = cv2.resize(bev_colored, (target_width, target_height))
        
        # Create frame
        frame = np.hstack([input_resized, bev_resized])
        
        # Add frame counter
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Frame {frame_count + 1}/{len(bev_files)}', 
                   (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, 'Input', (10, target_height - 10), 
                   font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, 'BEV Prediction', (target_width + 10, target_height - 10), 
                   font, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{len(bev_files)} frames")
    
    out.release()
    print(f"\nVideo saved: {output_video}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count / fps:.1f} seconds")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Create video from MonoLayout predictions')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--bev_dir', type=str, required=True,
                        help='Directory containing BEV predictions')
    parser.add_argument('--output', type=str, default='monolayout_results.mp4',
                        help='Output video file')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames')
    
    args = parser.parse_args()
    
    create_video_from_predictions(args.input_dir, args.bev_dir, 
                                  args.output, args.fps, args.max_frames)


if __name__ == '__main__':
    main()
