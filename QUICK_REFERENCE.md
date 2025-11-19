# Argoverse 2 BEV Ground Truth - Quick Reference

## 📁 File Structure
```
monolayout_MR_M25_IIITH/
├── preprocessing/argoverse2/
│   └── generate_groundtruth_av2.py    # Main GT generation script
├── visualize_av2_bev.py               # Visualization script
├── create_av2_video.py                # Video creation from visualizations
├── ARGOVERSE2_BEV_PREPROCESSING_REPORT.txt  # Full technical report
└── argoverse2_visualizations/         # Output visualizations
```

## 🚀 Quick Start Commands

### Generate Road BEV GT
```bash
python preprocessing/argoverse2/generate_groundtruth_av2.py \
    --base_path /ssd_scratch/cvit/varunp/argoverse \
    --seg_class road \
    --split train \
    --max_logs 2
```

### Generate Vehicle BEV GT
```bash
python preprocessing/argoverse2/generate_groundtruth_av2.py \
    --base_path /ssd_scratch/cvit/varunp/argoverse \
    --seg_class vehicle \
    --split train \
    --max_logs 2
```

### Create Visualizations
```bash
python visualize_av2_bev.py \
    --base_path /ssd_scratch/cvit/varunp/argoverse \
    --split train \
    --output_dir ./argoverse2_visualizations \
    --max_logs 2
```

### Create Video
```bash
python create_av2_video.py \
    --vis_dir argoverse2_visualizations/{log_id}/ \
    --output sequence.mp4 \
    --fps 10
```

## 📊 Key Transformations

### Ego → BEV Coordinates
```python
# Given ego position (x_ego, y_ego)
bev_u = (y_ego + 20.0) / 0.15625  # Horizontal
bev_v = (40.0 - x_ego) / 0.15625  # Vertical (inverted)
```

### Ego → Camera Projection
```python
# Using AV2 API
pinhole_camera = dataloader.get_log_pinhole_camera(log_id, camera_name)
uv, _, is_valid = pinhole_camera.project_ego_to_img(ego_position)
```

## 🎨 Color Coding
- 🟢 **Green**: Vehicle on road in BEV range
- 🟠 **Orange**: Vehicle off road in BEV range
- 🔴 **Red**: Vehicle outside BEV range

## ✅ Vehicle Filtering Criteria
1. **Spatial**: 0-40m forward, ±20m lateral
2. **Camera Visibility**: Must be in camera FOV
3. **Road Overlap**: Must touch road lanes

## 📈 Results (2 Sequences)
- **Total Frames**: 638
- **Road BEV GT**: 638/638 (100%)
- **Vehicle BEV GT**: 632/638 (99.1%)
- **Videos**: 2 × 31.9 seconds @ 10 FPS

## 🔧 Key Parameters
```python
BEV_RANGE = 40.0        # meters
BEV_SIZE = 256          # pixels
RESOLUTION = 0.15625    # m/pixel
TIMESTAMP_THRESHOLD = 100  # milliseconds
ROAD_OVERLAP_RADIUS = 3    # pixels
```

## 📝 Output Locations
- **Road GT**: `{base_path}/train/{log_id}/road_gt/ring_front_center_{timestamp}.png`
- **Vehicle GT**: `{base_path}/train/{log_id}/car_bev_gt/ring_front_center_{timestamp}.jpg`
- **Visualizations**: `./argoverse2_visualizations/{log_id}/vis_{log_id}_{timestamp}.png`

## 🐛 Common Issues & Fixes

### Issue: Empty vehicle BEV
**Cause**: No vehicles passed all filters  
**Solution**: This is expected - some frames have no valid vehicles

### Issue: Vehicles off-road in BEV
**Cause**: Incorrect coordinate transformation  
**Fix**: Use AV2 API, annotations already in ego frame

### Issue: Ego arrow pointing wrong direction
**Cause**: Transforming vector instead of positions  
**Fix**: Transform positions first, then compute movement

## 📚 Documentation
- Full Technical Report: `ARGOVERSE2_BEV_PREPROCESSING_REPORT.txt`
- Summary: `ARGOVERSE2_BEV_GT_SUMMARY.md`
- This Reference: `QUICK_REFERENCE.md`
