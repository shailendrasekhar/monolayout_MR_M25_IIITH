#!/usr/bin/env python3
"""
Analyze BEV predictions - compute statistics
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json


def analyze_bev(bev_path):
    """Analyze a single BEV prediction"""
    bev = cv2.imread(str(bev_path), cv2.IMREAD_GRAYSCALE)
    
    if bev is None:
        return None
    
    total_pixels = bev.shape[0] * bev.shape[1]
    occupied_pixels = np.sum(bev > 128)  # Count white pixels
    
    return {
        'file': bev_path.name,
        'total_pixels': int(total_pixels),
        'occupied_pixels': int(occupied_pixels),
        'occupied_percentage': float(occupied_pixels / total_pixels * 100),
        'shape': bev.shape
    }


def batch_analyze(bev_dir, output_json=None):
    """Analyze all BEV predictions in a directory"""
    bev_dir = Path(bev_dir)
    bev_files = sorted(list(bev_dir.glob('*.png')))
    
    print(f"Analyzing {len(bev_files)} BEV predictions...")
    
    results = []
    occupied_percentages = []
    
    for bev_file in bev_files:
        result = analyze_bev(bev_file)
        if result:
            results.append(result)
            occupied_percentages.append(result['occupied_percentage'])
    
    # Compute statistics
    stats = {
        'total_images': len(results),
        'avg_occupied_percentage': float(np.mean(occupied_percentages)),
        'std_occupied_percentage': float(np.std(occupied_percentages)),
        'min_occupied_percentage': float(np.min(occupied_percentages)),
        'max_occupied_percentage': float(np.max(occupied_percentages)),
        'median_occupied_percentage': float(np.median(occupied_percentages)),
        'results': results
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BEV Analysis Summary")
    print(f"{'='*60}")
    print(f"Total images analyzed: {stats['total_images']}")
    print(f"Average occupied area: {stats['avg_occupied_percentage']:.2f}%")
    print(f"Std deviation: {stats['std_occupied_percentage']:.2f}%")
    print(f"Min occupied: {stats['min_occupied_percentage']:.2f}%")
    print(f"Max occupied: {stats['max_occupied_percentage']:.2f}%")
    print(f"Median occupied: {stats['median_occupied_percentage']:.2f}%")
    print(f"{'='*60}")
    
    # Save to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nDetailed results saved to: {output_json}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze MonoLayout BEV predictions')
    parser.add_argument('--bev_dir', type=str, required=True,
                        help='Directory containing BEV predictions')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save detailed results to JSON file')
    
    args = parser.parse_args()
    
    batch_analyze(args.bev_dir, args.output_json)


if __name__ == '__main__':
    main()
