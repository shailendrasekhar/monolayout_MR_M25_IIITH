# Argoverse 2 BEV Ground Truth Generation - Summary

## ✅ COMPLETED FIXES

### 1. **Coordinate System Understanding**
- **Confirmed AV2 Convention**: `+X=Forward`, `+Y=Left`, `+Z=Up`
- **Updated BEV Mapping**: 
  - AV2 X (forward) → BEV Y-axis (top=far, bottom=near)
  - AV2 Y (left) → BEV X-axis (left=left, right=right)

### 2. **Annotation Coordinate Frame**
- **Key Discovery**: Annotations are **ALREADY in ego frame**
- **Fix**: Removed incorrect city-to-ego transformation
- **Impact**: Vehicle positions now correctly interpreted

### 3. **Vehicle Corner Calculation**
- **Bug**: Was using `[X±width, Y±length]`
- **Fix**: Changed to `[X±length, Y±width]`
- **Reason**: X=Forward (length), Y=Left (width)

### 4. **Visualization with AV2 API**
- **Method**: `pinhole_camera.project_ego_to_img()`
- **Proper Handling**: Check `is_valid` flag from projection result
- **Result**: 13 vehicles projected correctly per image

### 5. **Timestamp Loop Fix**
- **Bug**: BEV generation looped through annotation timestamps, missing some camera images
- **Fix**: Changed to loop through camera timestamps instead
- **Impact**: Now generates BEV for ALL camera images (5/5 instead of 3/5)

### 6. **Ego Motion Arrow Fix**
- **Bug**: Arrow pointed left/back instead of forward
- **Fix**: Corrected coordinate transformation (transform positions, not vectors) and BEV mapping
- **Impact**: Arrow now correctly shows ego vehicle's forward motion direction

### 7. **Vehicle-Road Filtering and Color Coding**
- **Feature**: Color code vehicles based on road overlap
- **Colors**: 
  - **Green**: Vehicles on road in BEV range
  - **Orange**: Vehicles off road in BEV range  
  - **Red**: Vehicles outside BEV range
- **Filtering**: Vehicles not touching roads are excluded from visualization
- **Method**: Check BEV coordinate overlap with road GT

## 📊 RESULTS

### Ground Truth Generation
- **Vehicle GT**: 5/5 images with content ✅ (FIXED: was 3/5)
- **Road GT**: 5/5 images with content ✅
- **Vehicle-road overlap**: 44.38% ✅
- **Center distance**: 10.0 pixels (improved from 35 pixels)
- **Coverage**: BEV generated for ALL camera timestamps ✅

### Visualization Quality
- **Vehicles projected**: 13 per image using AV2 API
- **BEV vehicles**: 6 per image (within 40m x 40m range)
- **Bounding boxes**: Properly aligned with actual vehicles
- **Color coding**: Green=On Road, Orange=Off Road, Red=Outside BEV
- **Road filtering**: Only shows vehicles touching road lanes
- **Ego arrow**: Correctly points in forward direction ✅

## 📁 KEY FILES

### Generation Scripts
- `preprocessing/argoverse2/generate_groundtruth_av2.py` - Main BEV GT generation
  - Fixed vehicle corner calculation
  - Correct coordinate system usage
  - Proper annotation handling

### Visualization Scripts
- `visualize_av2_bev.py` - Side-by-side visualization (camera + vehicle BEV + road BEV)
  - Uses AV2 API for bounding box projection
  - Shows 13 vehicles with proper alignment
  
- `use_av2_project_ego_to_img.py` - Standalone bounding box visualization
  - Demonstrates correct AV2 API usage
  - Reference implementation for projection

### Debug/Analysis Scripts
- `verify_coordinate_convention.py` - Confirmed AV2 coordinate system
- `analyze_annotation_coordinates.py` - Analyzed annotation coordinate frames
- `detailed_alignment_check.py` - Vehicle-road overlap analysis

## 🎯 USAGE

### Generate BEV Ground Truth

```bash
# Generate vehicle BEV GT
python preprocessing/argoverse2/generate_groundtruth_av2.py \
    --base_path /path/to/argoverse \
    --seg_class vehicle \
    --split train

# Generate road BEV GT
python preprocessing/argoverse2/generate_groundtruth_av2.py \
    --base_path /path/to/argoverse \
    --seg_class road \
    --split train
```

### Create Visualizations

```bash
# Create side-by-side visualizations
python visualize_av2_bev.py \
    --base_path /path/to/argoverse \
    --split train \
    --output_dir ./visualizations \
    --max_logs 5 \
    --max_images 10
```

### Verify Alignment

```bash
# Check vehicle-road overlap
python detailed_alignment_check.py
```

## 🔑 KEY INSIGHTS

1. **Annotations are in ego frame** - No transformation needed
2. **AV2 uses +X=Forward** - Different from typical robotics convention
3. **Use AV2 API for projection** - Don't implement manual transformations
4. **Check projection validity** - `is_valid` flag indicates if point is in view
5. **Timestamp synchronization** - Use lidar timestamps for annotations

## 📈 NEXT STEPS

1. **Generate full dataset**: Remove `--max_logs` limit
2. **Train MonoLayout**: Use corrected BEV ground truth
3. **Evaluate results**: Compare with baseline performance
4. **Fine-tune parameters**: Adjust BEV range if needed (currently 40m x 40m)

## 🐛 DEBUGGING TIPS

If bounding boxes don't align:
1. Check that annotations are used directly (no transformation)
2. Verify AV2 API is returning valid projections (`is_valid=True`)
3. Confirm image dimensions match camera calibration
4. Check timestamp synchronization (should be <100ms apart)

If BEV GT is empty:
1. Verify coordinate bounds match AV2 convention
2. Check that vehicles/roads are within BEV range
3. Ensure `world_to_bev()` mapping is correct

## 📚 REFERENCES

- Argoverse 2 API Documentation
- AV2 Coordinate System: +X=Forward, +Y=Left, +Z=Up
- PinholeCamera.project_ego_to_img() method
- BEV Range: X=[0,40]m (forward), Y=[-20,20]m (left-right)
