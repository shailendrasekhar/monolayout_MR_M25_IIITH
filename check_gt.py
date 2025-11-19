import os
import cv2
import numpy as np

# Check the updated vehicle GT
base_path = '/ssd_scratch/cvit/varunp/argoverse/train'
log_id = '1842383a-1577-3b7a-90db-41a9a6668ee2'
car_dir = os.path.join(base_path, log_id, 'car_bev_gt')

if os.path.exists(car_dir):
    files = sorted(os.listdir(car_dir))[:10]  # Check first 10 files
    print(f'Vehicle GT files: {len(os.listdir(car_dir))} total')
    
    non_empty_count = 0
    for f in files:
        img_path = os.path.join(car_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            unique_vals = np.unique(img)
            has_content = len(unique_vals) > 1 or (len(unique_vals) == 1 and unique_vals[0] != 0)
            if has_content:
                non_empty_count += 1
                print(f'{f}: min={img.min()}, max={img.max()}, unique={len(unique_vals)} - HAS CONTENT')
            else:
                print(f'{f}: min={img.min()}, max={img.max()}, unique={len(unique_vals)} - empty')
    
    print(f'Non-empty images: {non_empty_count}/{len(files)}')
