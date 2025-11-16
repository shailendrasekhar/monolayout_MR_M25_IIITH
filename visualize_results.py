#!/usr/bin/env python3
"""
Visualization Tool for MonoLayout Results
Creates side-by-side comparisons of input images and BEV predictions
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def create_side_by_side(input_img_path, bev_img_path, output_path, overlay=False):
    """
    Create side-by-side visualization of input and BEV prediction
    
    Args:
        input_img_path: Path to original camera image
        bev_img_path: Path to BEV prediction
        output_path: Path to save visualization
        overlay: If True, also create an overlay visualization
    """
    # Read images
    input_img = cv2.imread(input_img_path)
    bev_img = cv2.imread(bev_img_path, cv2.IMREAD_GRAYSCALE)
    
    if input_img is None:
        print(f"Error: Could not read {input_img_path}")
        return False
    if bev_img is None:
        print(f"Error: Could not read {bev_img_path}")
        return False
    
    # Resize input image to match width
    target_height = 512
    target_width = 1024
    input_resized = cv2.resize(input_img, (target_width, target_height))
    
    # Create colored BEV (green for road/vehicle)
    # Use COLORMAP_SPRING for green-ish color (compatible with older OpenCV)
    bev_colored = cv2.applyColorMap(bev_img, cv2.COLORMAP_SPRING)
    bev_resized = cv2.resize(bev_colored, (target_width, target_height))
    
    # Create side-by-side
    combined = np.hstack([input_resized, bev_resized])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Input Image', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'BEV Prediction', (target_width + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")
    
    # Create overlay if requested
    if overlay:
        overlay_path = output_path.replace('.png', '_overlay.png')
        
        # Resize BEV to match input
        bev_small = cv2.resize(bev_colored, (input_img.shape[1] // 4, input_img.shape[0] // 4))
        
        # Create overlay in bottom right corner
        overlay_img = input_img.copy()
        h, w = bev_small.shape[:2]
        overlay_img[-h:, -w:] = cv2.addWeighted(overlay_img[-h:, -w:], 0.3, bev_small, 0.7, 0)
        
        cv2.imwrite(overlay_path, overlay_img)
        print(f"Saved overlay: {overlay_path}")
    
    return True


def batch_visualize(input_dir, bev_dir, output_dir, max_images=None, overlay=False):
    """
    Create visualizations for all images in a directory
    """
    input_dir = Path(input_dir)
    bev_dir = Path(bev_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all BEV predictions
    bev_files = list(bev_dir.glob('*.png'))
    
    if max_images:
        bev_files = bev_files[:max_images]
    
    print(f"Found {len(bev_files)} BEV predictions")
    
    success_count = 0
    for bev_file in bev_files:
        # Find corresponding input image
        input_file = input_dir / (bev_file.stem + '.jpg')
        if not input_file.exists():
            input_file = input_dir / (bev_file.stem + '.png')
        
        if not input_file.exists():
            print(f"Warning: No input image found for {bev_file.name}")
            continue
        
        output_file = output_dir / f"vis_{bev_file.name}"
        
        if create_side_by_side(str(input_file), str(bev_file), str(output_file), overlay):
            success_count += 1
    
    print(f"\nCreated {success_count} visualizations in {output_dir}")
    return success_count


def main():
    parser = argparse.ArgumentParser(description='Visualize MonoLayout predictions')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--bev_dir', type=str, required=True,
                        help='Directory containing BEV predictions')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save visualizations')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--overlay', action='store_true',
                        help='Also create overlay visualizations')
    
    args = parser.parse_args()
    
    batch_visualize(args.input_dir, args.bev_dir, args.output_dir, 
                   args.max_images, args.overlay)


if __name__ == '__main__':
    main()
