#!/usr/bin/env python3
"""
Merge static and dynamic BEV predictions into a combined colored BEV.
Static (roads) -> Green channel
Dynamic (vehicles) -> Red channel
Overlap -> Yellow (R+G)

Saves merged images to out_dir/both/
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def merge_bevs(static_path, dynamic_path, out_path):
    static = cv2.imread(str(static_path), cv2.IMREAD_GRAYSCALE)
    dynamic = cv2.imread(str(dynamic_path), cv2.IMREAD_GRAYSCALE)

    if static is None and dynamic is None:
        return False

    # Determine output size (use static if available else dynamic)
    ref = static if static is not None else dynamic
    h, w = ref.shape

    # Resize inputs to reference if sizes mismatch
    if static is not None and static.shape != (h, w):
        static = cv2.resize(static, (w, h))
    if dynamic is not None and dynamic.shape != (h, w):
        dynamic = cv2.resize(dynamic, (w, h))

    # Create color image
    merged = np.zeros((h, w, 3), dtype=np.uint8)

    # Threshold masks
    if static is not None:
        static_mask = (static > 128)
        merged[static_mask, 1] = 255  # green channel
    else:
        static_mask = np.zeros((h, w), dtype=bool)

    if dynamic is not None:
        dynamic_mask = (dynamic > 128)
        merged[dynamic_mask, 2] = 255  # red channel (OpenCV BGR: index 2 is red?)
        # Note: OpenCV uses BGR order; indices: 0=B,1=G,2=R
    else:
        dynamic_mask = np.zeros((h, w), dtype=bool)

    # If both masks true, both channels will be set -> yellow (R+G)

    # Also create a grayscale merged occupancy map for optional analysis:
    occ = np.zeros((h, w), dtype=np.uint8)
    occ[static_mask] = 1
    occ[dynamic_mask] = 2
    occ[(static_mask) & (dynamic_mask)] = 2

    # Save merged color and occupancy
    cv2.imwrite(str(out_path), merged)
    occ_path = str(out_path).replace('.png', '_occ.png')
    cv2.imwrite(occ_path, (occ * 127).astype(np.uint8))

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static_dir', required=True)
    parser.add_argument('--dynamic_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--max_images', type=int, default=None)
    args = parser.parse_args()

    static_dir = Path(args.static_dir)
    dynamic_dir = Path(args.dynamic_dir)
    out_dir = Path(args.out_dir) / 'both'
    out_dir.mkdir(parents=True, exist_ok=True)

    static_files = sorted([p for p in static_dir.glob('*.png')])
    dynamic_files = sorted([p for p in dynamic_dir.glob('*.png')])

    # Build mapping by stem
    dynamic_map = {p.stem: p for p in dynamic_files}

    count = 0
    for s in static_files:
        stem = s.stem
        d = dynamic_map.get(stem)
        out_path = out_dir / f"{stem}.png"
        if d is None:
            # No dynamic match: still merge using empty dynamic
            merge_bevs(s, None, out_path)
            count += 1
        else:
            merge_bevs(s, d, out_path)
            count += 1
        if args.max_images and count >= args.max_images:
            break

    # Also add any dynamic files missing in static
    for stem, d in dynamic_map.items():
        out_path = out_dir / f"{stem}.png"
        if out_path.exists():
            continue
        merge_bevs(None, d, out_path)
        count += 1
        if args.max_images and count >= args.max_images:
            break

    print(f"Merged {count} BEV images -> {out_dir}")

if __name__ == '__main__':
    main()
