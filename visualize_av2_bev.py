#!/usr/bin/env python3
"""
Visualization Tool for Argoverse 2 BEV Ground Truth
Creates side-by-side comparisons of input camera images and generated BEV ground truth
"""

import cv2
import numpy as np
import os
import argparse
import sys
from pathlib import Path

# Add AV2 API to path
sys.path.append('/home2/varun.paturkar/miniconda3/envs/monolayout/lib/python3.10/site-packages')

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.utils.io import read_feather


def check_vehicle_on_road(vehicle_ego_pos, road_bev, bev_size=256, bev_range=40):
    """Check if a vehicle position overlaps with road in BEV."""
    if road_bev is None:
        return False
    
    # Convert ego coordinates to BEV image coordinates
    # AV2: +X=Forward, +Y=Left
    # BEV: X-axis=horizontal (left-right), Y-axis=vertical (top-bottom)
    av2_x_forward = vehicle_ego_pos[0]
    av2_y_left = vehicle_ego_pos[1]
    
    # Check if in BEV range
    if not (0 <= av2_x_forward <= bev_range and -bev_range/2 <= av2_y_left <= bev_range/2):
        return False
    
    # Map to BEV image coordinates
    res = bev_range / float(bev_size)
    bev_x = int((av2_y_left + bev_range/2) / res)  # Y (left) -> X (horizontal)
    bev_y = int((bev_range - av2_x_forward) / res)  # X (forward) -> Y (inverted vertical)
    
    # Check bounds
    if not (0 <= bev_x < bev_size and 0 <= bev_y < bev_size):
        return False
    
    # Check if there's road at this position (with small radius for tolerance)
    radius = 3
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            check_x = bev_x + dx
            check_y = bev_y + dy
            if 0 <= check_x < bev_size and 0 <= check_y < bev_size:
                if road_bev[check_y, check_x] > 0:  # Road pixel
                    return True
    
    return False


def draw_bounding_boxes_on_camera(camera_img, log_id, timestamp, base_path):
    """Draw 3D bounding boxes projected onto camera image using AV2 API."""
    try:
        # Initialize dataloader
        split = "train"  # Assuming train split
        camera_name = "ring_front_center"
        
        dataloader = AV2SensorDataLoader(
            data_dir=Path(base_path) / split,
            labels_dir=Path(base_path) / split
        )
        
        # Get lidar timestamps
        lidar_dir = os.path.join(base_path, split, log_id, "sensors", "lidar")
        if not os.path.exists(lidar_dir):
            return camera_img
            
        lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.feather')])
        lidar_timestamps = [int(f.replace('.feather', '')) for f in lidar_files]
        
        # Find closest lidar timestamp to camera timestamp
        camera_timestamp = int(timestamp)
        closest_lidar_ts = min(lidar_timestamps, key=lambda x: abs(x - camera_timestamp))
        
        # Get annotations at lidar timestamp using AV2 API
        annotations = dataloader.get_labels_at_lidar_timestamp(log_id, closest_lidar_ts)
        if annotations is None or len(annotations) == 0:
            return camera_img
        
        # Filter for vehicles
        vehicle_annotations = [ann for ann in annotations if ann.category in 
                             ['REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER']]
        
        if len(vehicle_annotations) == 0:
            return camera_img
        
        # Draw actual bounding boxes using AV2 API projection
        img_with_boxes = camera_img.copy()
        img_height, img_width = img_with_boxes.shape[:2]
        
        # Add text overlay showing number of vehicles
        cv2.putText(img_with_boxes, f"Total Vehicles: {len(vehicle_annotations)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Get pinhole camera object from AV2 API
        pinhole_camera = dataloader.get_log_pinhole_camera(log_id, camera_name)
        if pinhole_camera is None:
            return camera_img
        
        # Load road BEV to check if vehicles are on roads
        road_bev = None
        try:
            road_bev_path = os.path.join(base_path, split, log_id, "road_gt", f"{camera_name}_{camera_timestamp}.png")
            if os.path.exists(road_bev_path):
                road_bev = cv2.imread(road_bev_path, cv2.IMREAD_GRAYSCALE)
        except:
            pass
        
        vehicles_projected = 0
        vehicles_in_bev = 0
        vehicles_on_road = 0
        
        # Process each vehicle using AV2 API projection
        for i, annotation in enumerate(vehicle_annotations):
            try:
                # Get vehicle position - annotations are ALREADY in ego frame
                vehicle_ego_pos = np.array([
                    annotation.dst_SE3_object.translation[0],
                    annotation.dst_SE3_object.translation[1],
                    annotation.dst_SE3_object.translation[2]
                ])
                
                # Check if in BEV range
                in_bev_range = (0 <= vehicle_ego_pos[0] <= 40 and -20 <= vehicle_ego_pos[1] <= 20)
                if in_bev_range:
                    vehicles_in_bev += 1
                
                # Check if vehicle is on road
                on_road = check_vehicle_on_road(vehicle_ego_pos, road_bev)
                
                # For visualization: show all vehicles with color coding
                # For BEV GT generation: only include vehicles on roads
                
                # Use AV2 API to project ego point to camera image
                uv_cam = pinhole_camera.project_ego_to_img(vehicle_ego_pos.reshape(1, -1))
                
                # uv_cam returns (coordinates, homogeneous_coords, is_valid_array)
                # is_valid[0] = True means the point is in front of camera and within FOV
                if uv_cam is not None and len(uv_cam) >= 3:
                    coords, _, is_valid = uv_cam
                    
                    # Only draw if projection is valid (in camera view)
                    if len(coords) > 0 and len(is_valid) > 0 and is_valid[0]:
                        x_img, y_img = coords[0]
                        
                        # Double-check within image bounds (should already be true if is_valid)
                        if 0 <= x_img < img_width and 0 <= y_img < img_height:
                            vehicles_projected += 1
                            if on_road:
                                vehicles_on_road += 1
                            
                            # Color code based on BEV range and road status
                            if in_bev_range and on_road:
                                color = (0, 255, 0)  # Green for BEV vehicles on road
                                radius = 12
                                thickness = 3
                            elif in_bev_range and not on_road:
                                color = (0, 165, 255)  # Orange for BEV vehicles off road
                                radius = 10
                                thickness = 2
                            else:
                                color = (0, 0, 255)  # Red for non-BEV vehicles
                                radius = 8
                                thickness = 2
                            
                            # Draw circle marker
                            cv2.circle(img_with_boxes, (int(x_img), int(y_img)), radius, color, -1)
                            
                            # Draw bounding box
                            distance = np.linalg.norm(vehicle_ego_pos)
                            box_size = max(25, int(150 / distance))
                            
                            x1, y1 = int(x_img - box_size), int(y_img - box_size)
                            x2, y2 = int(x_img + box_size), int(y_img + box_size)
                            
                            # Ensure within bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(img_width - 1, x2)
                            y2 = min(img_height - 1, y2)
                            
                            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
                            
                            # Add label
                            label = f"V{i+1}:{distance:.0f}m"
                            if in_bev_range and on_road:
                                label += "(BEV-Road)"
                            elif in_bev_range:
                                label += "(BEV-OffRoad)"
                            
                            cv2.putText(img_with_boxes, label, (int(x_img) + 15, int(y_img) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            except Exception as e:
                continue
        
        # Add summary at bottom
        summary_text = f"Projected: {vehicles_projected} | On Road: {vehicles_on_road} | In BEV: {vehicles_in_bev}"
        cv2.putText(img_with_boxes, summary_text, (10, img_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add legend
        cv2.putText(img_with_boxes, "Green=On Road | Orange=Off Road | Red=Outside BEV", (10, img_height - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_with_boxes
            
    except Exception as e:
        print(f"Error drawing bounding boxes: {e}")
        return camera_img


def create_av2_visualization(camera_img_path, vehicle_bev_path, road_bev_path, output_path, log_id=None, timestamp=None, base_path=None):
    """
    Create visualization of camera image with vehicle and road BEV ground truth
    
    Args:
        camera_img_path: Path to camera image
        vehicle_bev_path: Path to vehicle BEV ground truth
        road_bev_path: Path to road BEV ground truth  
        output_path: Path to save visualization
        log_id: Log ID for ego motion calculation
        timestamp: Timestamp for ego motion calculation
    """
    # Read images
    camera_img = cv2.imread(camera_img_path)
    vehicle_bev = cv2.imread(vehicle_bev_path, cv2.IMREAD_GRAYSCALE) if vehicle_bev_path and os.path.exists(vehicle_bev_path) else None
    road_bev = cv2.imread(road_bev_path, cv2.IMREAD_GRAYSCALE) if road_bev_path and os.path.exists(road_bev_path) else None
    
    if camera_img is None:
        print(f"Error: Could not read camera image {camera_img_path}")
        return False
    
    # Draw bounding boxes on camera image if we have the necessary data
    if log_id and timestamp and base_path:
        camera_img = draw_bounding_boxes_on_camera(camera_img, log_id, timestamp, base_path)
    
    # Target dimensions
    target_height = 400
    target_width = 400
    
    # Resize camera image
    camera_resized = cv2.resize(camera_img, (target_width, target_height))
    
    # Process vehicle BEV
    if vehicle_bev is not None:
        # Create colored vehicle BEV (red for vehicles)
        vehicle_colored = np.zeros((vehicle_bev.shape[0], vehicle_bev.shape[1], 3), dtype=np.uint8)
        vehicle_colored[:, :, 2] = vehicle_bev  # Red channel
        vehicle_resized = cv2.resize(vehicle_colored, (target_width, target_height))
    else:
        vehicle_resized = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Process road BEV
    if road_bev is not None:
        # Create colored road BEV (green for roads)
        road_colored = np.zeros((road_bev.shape[0], road_bev.shape[1], 3), dtype=np.uint8)
        road_colored[:, :, 1] = road_bev  # Green channel
        road_resized = cv2.resize(road_colored, (target_width, target_height))
    else:
        road_resized = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Create combined BEV (road + vehicles)
    combined_bev = cv2.addWeighted(road_resized, 0.7, vehicle_resized, 0.8, 0)
    
    # Add ego vehicle direction indicator if we have log_id and timestamp
    if log_id and timestamp:
        try:
            # Import AV2 modules
            import sys
            sys.path.append('/home2/varun.paturkar/miniconda3/envs/monolayout/lib/python3.10/site-packages')
            from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
            from av2.utils.io import read_feather
            
            # Get ego motion direction
            base_path = "/ssd_scratch/cvit/varunp/argoverse"
            split = "train"
            
            dataloader = AV2SensorDataLoader(
                data_dir=Path(base_path) / split,
                labels_dir=Path(base_path) / split
            )
            
            # Get ego poses around this timestamp
            ego_poses_path = Path(base_path) / split / log_id / "city_SE3_egovehicle.feather"
            ego_poses_df = read_feather(ego_poses_path)
            timestamps = sorted(ego_poses_df['timestamp_ns'].unique())
            
            # Find current timestamp index
            current_idx = None
            for i, ts in enumerate(timestamps):
                if ts == int(timestamp):
                    current_idx = i
                    break
            
            if current_idx is not None and current_idx > 0:
                # Get previous and current poses
                prev_timestamp = timestamps[current_idx - 1]
                curr_timestamp = timestamps[current_idx]
                
                prev_pose = dataloader.get_city_SE3_ego(log_id, prev_timestamp)
                curr_pose = dataloader.get_city_SE3_ego(log_id, curr_timestamp)
                
                if prev_pose is not None and curr_pose is not None:
                    # CORRECT METHOD: Transform positions, then calculate movement
                    # Transform previous position to current ego frame
                    prev_pos_3d = np.array([[prev_pose.translation[0], prev_pose.translation[1], prev_pose.translation[2]]])
                    prev_in_curr_ego = curr_pose.inverse().transform_point_cloud(prev_pos_3d)[0]
                    
                    # Current position in current ego frame is origin
                    curr_in_curr_ego = np.array([0, 0, 0])
                    
                    # Movement in ego frame (from previous to current)
                    movement_ego = curr_in_curr_ego[:2] - prev_in_curr_ego[:2]
                    
                    # Draw ego direction arrow on combined BEV
                    center_x, center_y = target_width // 2, target_height - 20  # Bottom center
                    
                    # Scale and rotate movement vector for display
                    if np.linalg.norm(movement_ego) > 0.01:  # Only if there's significant movement
                        arrow_length = 30
                        movement_norm = movement_ego / np.linalg.norm(movement_ego)
                        
                        # AV2: movement_ego[0]=X=Forward, movement_ego[1]=Y=Left
                        # BEV image: horizontal=left-right, vertical=top(far)-bottom(near)
                        # Map: AV2 X (forward) -> BEV up (negative Y), AV2 Y (left) -> BEV left (negative X)
                        arrow_end_x = int(center_x - movement_norm[1] * arrow_length)  # AV2 Y (left) -> image X (left)
                        arrow_end_y = int(center_y - movement_norm[0] * arrow_length)  # AV2 X (forward) -> image Y (up)
                        
                        # Draw arrow
                        cv2.arrowedLine(combined_bev, (center_x, center_y), (arrow_end_x, arrow_end_y), 
                                      (255, 255, 0), 3, tipLength=0.3)  # Cyan arrow
                        
                        # Add text
                        cv2.putText(combined_bev, 'EGO', (center_x - 15, center_y + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        except Exception as e:
            print(f"Could not add ego direction: {e}")
            pass
    
    # Create 2x2 grid: [Camera, Combined BEV]
    #                  [Road BEV, Vehicle BEV]
    top_row = np.hstack([camera_resized, combined_bev])
    bottom_row = np.hstack([road_resized, vehicle_resized])
    final_image = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    cv2.putText(final_image, 'Camera Image', (10, 25), font, font_scale, color, thickness)
    cv2.putText(final_image, 'Combined BEV', (target_width + 10, 25), font, font_scale, color, thickness)
    cv2.putText(final_image, 'Road GT', (10, target_height + 25), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(final_image, 'Vehicle GT', (target_width + 10, target_height + 25), font, font_scale, (0, 0, 255), thickness)
    
    # Add timestamp info
    timestamp = Path(camera_img_path).stem.split('_')[-1]
    cv2.putText(final_image, f'Timestamp: {timestamp}', (10, final_image.shape[0] - 10), 
                font, 0.5, (255, 255, 255), 1)
    
    # Save
    cv2.imwrite(output_path, final_image)
    print(f"Saved: {output_path}")
    return True


def visualize_av2_log(log_path, output_dir, max_images=10, base_path=None):
    """
    Visualize BEV ground truth for a single Argoverse 2 log
    
    Args:
        log_path: Path to the log directory
        output_dir: Directory to save visualizations
        max_images: Maximum number of images to process
        base_path: Base path to the Argoverse dataset
    """
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    camera_dir = log_path / "sensors" / "cameras" / "ring_front_center"
    vehicle_bev_dir = log_path / "car_bev_gt"
    road_bev_dir = log_path / "road_gt"
    
    if not camera_dir.exists():
        print(f"Error: Camera directory not found: {camera_dir}")
        return 0
    
    # Get camera images
    camera_files = sorted([f for f in camera_dir.glob('*.jpg')])
    
    if max_images:
        camera_files = camera_files[:max_images]
    
    print(f"Processing {len(camera_files)} images from log {log_path.name}")
    
    success_count = 0
    for camera_file in camera_files:
        timestamp = camera_file.stem
        
        # Find corresponding BEV files
        vehicle_bev_file = vehicle_bev_dir / f"ring_front_center_{timestamp}.jpg"
        road_bev_file = road_bev_dir / f"ring_front_center_{timestamp}.png"
        
        # Create output filename
        output_file = output_dir / f"vis_{log_path.name}_{timestamp}.png"
        
        # Create visualization
        if create_av2_visualization(
            str(camera_file), 
            str(vehicle_bev_file) if vehicle_bev_file.exists() else None,
            str(road_bev_file) if road_bev_file.exists() else None,
            str(output_file),
            log_path.name,  # log_id
            timestamp,      # timestamp
            base_path       # base_path
        ):
            success_count += 1
    
    print(f"Created {success_count} visualizations for log {log_path.name}")
    return success_count


def visualize_multiple_logs(base_path, split, output_dir, max_logs=None, max_images_per_log=5):
    """
    Visualize BEV ground truth for multiple Argoverse 2 logs
    """
    base_path = Path(base_path)
    split_path = base_path / split
    output_dir = Path(output_dir)
    
    if not split_path.exists():
        print(f"Error: Split path not found: {split_path}")
        return 0
    
    # Get all log directories
    log_dirs = [d for d in split_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')]
    
    if max_logs:
        log_dirs = log_dirs[:max_logs]
    
    print(f"Processing {len(log_dirs)} logs from {split} split")
    
    total_success = 0
    for log_dir in log_dirs:
        log_output_dir = output_dir / log_dir.name
        success = visualize_av2_log(log_dir, log_output_dir, max_images_per_log, base_path)
        total_success += success
    
    print(f"\nTotal: Created {total_success} visualizations in {output_dir}")
    return total_success


def main():
    parser = argparse.ArgumentParser(description='Visualize Argoverse 2 BEV Ground Truth')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Path to Argoverse 2 dataset root')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save visualizations')
    parser.add_argument('--log_id', type=str, default=None,
                        help='Specific log ID to visualize (optional)')
    parser.add_argument('--max_logs', type=int, default=3,
                        help='Maximum number of logs to process')
    parser.add_argument('--max_images', type=int, default=5,
                        help='Maximum number of images per log')
    
    args = parser.parse_args()
    
    if args.log_id:
        # Visualize specific log
        log_path = Path(args.base_path) / args.split / args.log_id
        visualize_av2_log(log_path, args.output_dir, args.max_images)
    else:
        # Visualize multiple logs
        visualize_multiple_logs(args.base_path, args.split, args.output_dir, 
                               args.max_logs, args.max_images)


if __name__ == '__main__':
    main()
