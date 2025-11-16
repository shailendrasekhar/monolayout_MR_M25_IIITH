# What Can You Do With Test Images & Results? 🎨

## Available Tools & Visualizations

I've created several powerful tools for you to analyze and visualize your MonoLayout results!

---

## 1. 📊 **Visualize Results (Side-by-Side Comparison)**

Create beautiful side-by-side comparisons of input images and BEV predictions:

### **Quick Test (1 image):**
```bash
conda activate monolayout

python3 visualize_results.py \
    --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/test_single/static/ \
    --output_dir ./visualizations/test/
```

### **Batch Process (Multiple images):**
```bash
python3 visualize_results.py \
    --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/argoverse_static/static/ \
    --output_dir ./visualizations/static/ \
    --max_images 50
```

### **With Overlay Effect:**
```bash
python3 visualize_results.py \
    --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/argoverse_static/static/ \
    --output_dir ./visualizations/static_overlay/ \
    --overlay \
    --max_images 20
```

**Output**: Side-by-side PNG images showing input and colored BEV prediction

---

## 2. 🎬 **Create Video from Results**

Generate a video showing your predictions over time:

### **Create Static (Roads) Video:**
```bash
conda activate monolayout

python3 create_video.py \
    --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/argoverse_static/static/ \
    --output ./videos/static_roads.mp4 \
    --fps 10 \
    --max_frames 100
```

### **Create Dynamic (Vehicles) Video:**
```bash
python3 create_video.py \
    --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/argoverse_dynamic/dynamic/ \
    --output ./videos/dynamic_vehicles.mp4 \
    --fps 10
```

### **Fast-Motion Video:**
```bash
python3 create_video.py \
    --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/argoverse_static/static/ \
    --output ./videos/fast_motion.mp4 \
    --fps 30
```

**Output**: MP4 video file showing predictions in sequence

---

## 3. 📈 **Analyze Predictions (Statistics)**

Get detailed statistics about your predictions:

### **Analyze Static Predictions:**
```bash
conda activate monolayout

python3 analyze_predictions.py \
    --bev_dir ./output/argoverse_static/static/ \
    --output_json ./analysis/static_analysis.json
```

### **Analyze Dynamic Predictions:**
```bash
python3 analyze_predictions.py \
    --bev_dir ./output/argoverse_dynamic/dynamic/ \
    --output_json ./analysis/dynamic_analysis.json
```

**Output**: Statistics like:
- Average occupied area percentage
- Min/max/median values
- Distribution of predictions
- JSON file with detailed per-image results

---

## 4. 🔄 **Compare Static vs Dynamic**

Create side-by-side comparison of both prediction types:

```bash
# First, run both inferences
python3 test.py --type static --model_path ./pretrained_models/argoverse_static/ \
    --image_path ./test_images/argoverse/ --out_dir ./output/both/ --ext jpg

python3 test.py --type dynamic --model_path ./pretrained_models/argoverse_dynamic/ \
    --image_path ./test_images/argoverse/ --out_dir ./output/both/ --ext jpg

# Then visualize both
python3 visualize_results.py --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/both/static/ --output_dir ./visualizations/comparison_static/

python3 visualize_results.py --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/both/dynamic/ --output_dir ./visualizations/comparison_dynamic/
```

---

## 5. 🎯 **Interactive Jupyter Notebook** (Coming Soon)

Create a Jupyter notebook to:
- Load and display results interactively
- Filter by prediction confidence
- Show specific time sequences
- Create custom visualizations

---

## 6. 📊 **Generate Report**

Create an HTML report with all results:

```bash
# Create visualizations first
python3 visualize_results.py --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/argoverse_static/static/ \
    --output_dir ./report/images/ --max_images 20

# Analyze predictions
python3 analyze_predictions.py --bev_dir ./output/argoverse_static/static/ \
    --output_json ./report/statistics.json

# Create video
python3 create_video.py --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/argoverse_static/static/ \
    --output ./report/demo_video.mp4 --fps 10 --max_frames 50
```

---

## 7. 🎨 **Custom Visualizations**

### **Extract Specific Frames:**
```bash
# Copy specific interesting frames
cp test_images/argoverse/ring_front_center_315978406032859416.jpg interesting_frames/
cp output/argoverse_static/static/ring_front_center_315978406032859416.png interesting_frames/

# Visualize just those
python3 visualize_results.py --input_dir ./interesting_frames/ \
    --bev_dir ./interesting_frames/ --output_dir ./interesting_viz/
```

### **Create Montage:**
```bash
# Use ImageMagick to create a grid
montage visualizations/static/vis_*.png -tile 4x4 -geometry 256x256+2+2 montage_grid.png
```

---

## 8. 🔬 **Research & Analysis Ideas**

### **Things you can investigate:**

1. **Road Coverage Analysis**
   - How much of the BEV space is predicted as road?
   - Does it vary by scene type?

2. **Temporal Consistency**
   - How consistent are predictions across sequential frames?
   - Can you detect sudden changes?

3. **Comparison with Ground Truth**
   - If you have ground truth, compute IoU scores
   - Identify where the model performs well/poorly

4. **Scene Understanding**
   - Urban vs highway scenes
   - Day vs night (if available)
   - Weather conditions

5. **Vehicle Detection Quality**
   - Count predicted vehicles per frame
   - Size and distribution of detections

---

## 🚀 **Quick Start Examples**

### **Example 1: Quick Visualization Test**
```bash
conda activate monolayout

# Visualize your existing single test result
python3 visualize_results.py \
    --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/test_single/static/ \
    --output_dir ./quick_viz/ \
    --overlay
```

### **Example 2: Create Demo Video (10 frames)**
```bash
# Run inference on first 10 images
python3 test.py --type static \
    --model_path ./pretrained_models/argoverse_static/ \
    --image_path ./test_images/argoverse/ \
    --out_dir ./output/demo/ --ext jpg

# Create video
python3 create_video.py \
    --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/demo/static/ \
    --output ./demo_video.mp4 \
    --fps 5 --max_frames 10
```

### **Example 3: Full Analysis Pipeline**
```bash
# 1. Run inference (if not done)
# 2. Analyze
python3 analyze_predictions.py \
    --bev_dir ./output/test_single/static/ \
    --output_json ./analysis.json

# 3. Visualize
python3 visualize_results.py \
    --input_dir ./test_images/argoverse/ \
    --bev_dir ./output/test_single/static/ \
    --output_dir ./final_viz/ \
    --overlay

# 4. View results
xdg-open ./final_viz/  # Opens file manager
```

---

## 📁 **Recommended Directory Structure**

After running all tools:
```
monolayout_MR_M25_IIITH/
├── test_images/
│   └── argoverse/           # Original images
├── output/
│   ├── argoverse_static/    # BEV predictions (roads)
│   └── argoverse_dynamic/   # BEV predictions (vehicles)
├── visualizations/
│   ├── static/              # Side-by-side images
│   ├── dynamic/
│   └── overlays/
├── videos/
│   ├── static_roads.mp4
│   └── dynamic_vehicles.mp4
├── analysis/
│   ├── static_analysis.json
│   └── dynamic_analysis.json
└── report/
    ├── images/
    ├── statistics.json
    └── demo_video.mp4
```

---

## 💡 **Tips & Tricks**

1. **Start Small**: Test on 5-10 images first before processing all 469
2. **Use max_frames**: Limit processing for quick tests
3. **Organize Output**: Create separate directories for different experiments
4. **Save Interesting Cases**: Keep examples of good/bad predictions
5. **Compare Models**: Run both static and dynamic, compare results

---

## ⚡ **Performance Notes**

- **Visualization**: Very fast (~0.1s per image)
- **Video Creation**: Fast (~0.2s per frame)
- **Analysis**: Very fast (~0.05s per image)

All visualization tools run on CPU and are much faster than inference!

---

Would you like me to run any of these examples for you right now?
