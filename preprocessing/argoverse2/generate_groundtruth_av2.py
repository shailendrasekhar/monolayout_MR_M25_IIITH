#!/usr/bin/env python3
"""
Argoverse 2 Ground Truth Generation for MonoLayout

This script generates bird's-eye-view ground truth layouts for both static (road) 
and dynamic (vehicle) elements from Argoverse 2 sensor dataset.

Key differences from Argoverse 1:
- Uses .feather files instead of JSON
- Different directory structure 
- Uses av2 API instead of argoverse API
- Different coordinate systems and camera naming
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Argoverse 2 imports
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import Cuboid
from av2.utils.io import read_feather
from av2.geometry.se3 import SE3


def get_args():
    parser = argparse.ArgumentParser(
        description="MonoLayout Argoverse 2 Data Preparation")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Path to the Argoverse 2 dataset root directory")
    parser.add_argument("--out_dir", type=str, default='',
                        help="Output directory to save layouts")
    parser.add_argument("--range", type=int, default=40,
                        help="Size of the rectangular grid in metric space")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="Occupancy map size")
    parser.add_argument(
        "--seg_class",
        type=str,
        choices=["road", "vehicle"],
        required=True,
        help="Generate layouts for road or vehicle")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"],
                        default="train", help="Dataset split to process")
    parser.add_argument("--camera", type=str, default="ring_front_center",
                        help="Camera to use for image correspondence")
    parser.add_argument("--max_logs", type=int, default=None,
                        help="Maximum number of logs to process (for testing)")
    
    return parser.parse_args()


class ArgoverseV2LayoutGenerator:
    """Generate BEV la 2 datyouts from Argoversea."""
    
    def __init__(self, args):
        self.args = args
        self.range = args.range
        self.occ_map_size = args.occ_map_size
        self.camera = args.camera
        
        # BEV parameters - CORRECTED for AV2 coordinate convention
        # AV2 ego frame: +X=Forward, +Y=Left, +Z=Up
        # BEV image: X-axis=horizontal (left-right), Y-axis=vertical (top-bottom)
        self.res = self.range / float(self.occ_map_size)
        
        # Map AV2 coordinates to BEV image coordinates:
        # AV2 X (forward) -> BEV Y-axis (top=far, bottom=near)
        # AV2 Y (left) -> BEV X-axis (left=left, right=right)
        self.x_forward_lb = 0           # 0m behind ego
        self.x_forward_ub = self.range  # 40m in front of ego
        self.y_left_lb = -self.range / 2  # 20m to the right
        self.y_left_ub = self.range / 2   # 20m to the left
        
        # Initialize dataloader
        self.dataloader = AV2SensorDataLoader(
            data_dir=Path(args.base_path) / args.split,
            labels_dir=Path(args.base_path) / args.split
        )
        
        print(f"Initialized AV2 dataloader with {len(self.dataloader.get_log_ids())} logs")
    
    def world_to_bev(self, points: np.ndarray) -> np.ndarray:
        """Convert AV2 ego coordinates to BEV image coordinates.
        
        AV2 ego frame: +X=Forward, +Y=Left, +Z=Up
        BEV image: X-axis=horizontal (left-right), Y-axis=vertical (top-bottom)
        
        Mapping:
        - AV2 X (forward) -> BEV Y-axis (inverted: top=far, bottom=near)  
        - AV2 Y (left) -> BEV X-axis (inverted: left=right, right=left)
        """
        av2_x_forward = points[:, 0]  # AV2 X = forward/backward
        av2_y_left = points[:, 1]     # AV2 Y = left/right
        
        # Map AV2 Y (left) to BEV X-axis (horizontal)
        # AV2 Y: left(+) to right(-) -> BEV X: left(0) to right(255)
        bev_x = ((-av2_y_left - self.y_left_lb) / (self.y_left_ub - self.y_left_lb) * self.occ_map_size).astype(np.int32)
        
        # Map AV2 X (forward) to BEV Y-axis (vertical)  
        # AV2 X: near(0) to far(40) -> BEV Y: bottom(255) to top(0)
        bev_y = ((self.x_forward_ub - av2_x_forward) / (self.x_forward_ub - self.x_forward_lb) * self.occ_map_size).astype(np.int32)
        
        # Clip to image bounds
        bev_x = np.clip(bev_x, 0, self.occ_map_size - 1)
        bev_y = np.clip(bev_y, 0, self.occ_map_size - 1)
        
        return np.column_stack([bev_x, bev_y])
    
    def generate_vehicle_bev(self, log_id: str, output_dir: str):
        """Generate vehicle BEV layouts for a single log."""
        print(f"Processing vehicle layouts for log: {log_id}")
        
        # Create output directory
        vehicle_dir = os.path.join(output_dir, log_id, "car_bev_gt")
        os.makedirs(vehicle_dir, exist_ok=True)
        
        # Get annotations
        annotations_path = os.path.join(self.args.base_path, self.args.split, log_id, "annotations.feather")
        if not os.path.exists(annotations_path):
            print(f"No annotations found for {log_id}")
            return
            
        annotations_df = read_feather(Path(annotations_path))
        
        # Get synchronized timestamps from annotations (these are lidar timestamps)
        unique_timestamps = sorted(annotations_df['timestamp_ns'].unique())
        
        # Get available camera timestamps
        camera_dir = os.path.join(self.args.base_path, self.args.split, log_id, "sensors", "cameras", self.camera)
        if not os.path.exists(camera_dir):
            print(f"Camera directory not found: {camera_dir}")
            return
            
        camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.jpg')])
        camera_timestamps = [int(f.replace('.jpg', '')) for f in camera_files]
        
        def find_closest_camera_timestamp(target_timestamp):
            """Find the closest camera timestamp to the target timestamp."""
            if not camera_timestamps:
                return None
            closest_idx = min(range(len(camera_timestamps)), 
                             key=lambda i: abs(camera_timestamps[i] - target_timestamp))
            return camera_timestamps[closest_idx]
        
        # Loop through camera timestamps instead of annotation timestamps
        # This ensures we generate BEV for every camera image
        for camera_ts in tqdm(camera_timestamps, desc=f"Processing {log_id} vehicle"):
            # Find closest annotation timestamp
            closest_annotation_ts = min(unique_timestamps, key=lambda x: abs(x - camera_ts))
                
            # Check if the timestamps are reasonably close (within 100ms)
            time_diff = abs(camera_ts - closest_annotation_ts) / 1e6  # Convert to milliseconds
            if time_diff > 100:  # Skip if more than 100ms apart
                continue
            
            # Get annotations for this timestamp
            frame_annotations = annotations_df[annotations_df['timestamp_ns'] == closest_annotation_ts]
            
            # Filter for vehicles
            vehicle_categories = ['REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER']
            vehicles = frame_annotations[frame_annotations['category'].isin(vehicle_categories)]
            
            # Create BEV image
            bev_image = np.zeros((self.occ_map_size, self.occ_map_size), dtype=np.uint8)
            
            # Load road BEV to filter vehicles not on roads
            road_bev = None
            try:
                road_bev_path = os.path.join(output_dir, log_id, "road_gt", f"{self.camera}_{camera_ts}.png")
                if os.path.exists(road_bev_path):
                    road_bev = cv2.imread(road_bev_path, cv2.IMREAD_GRAYSCALE)
            except:
                pass
            
            if len(vehicles) > 0:
                # Get ego pose for this timestamp to ensure consistent coordinate frame
                ego_pose = self.dataloader.get_city_SE3_ego(log_id, closest_annotation_ts)
                if ego_pose is None:
                    continue
                
                # Get pinhole camera to check if vehicles are visible in camera view
                pinhole_camera = self.dataloader.get_log_pinhole_camera(log_id, self.camera)
                if pinhole_camera is None:
                    print(f"   Warning: Could not get pinhole camera for {log_id}")
                    continue
                
                for _, vehicle in vehicles.iterrows():
                    # Vehicle annotations are in ego frame, but we need to ensure they're in the SAME ego frame
                    # as the one used for road generation. Transform vehicle from annotation ego frame to current ego frame.
                    vehicle_pos_annotation = np.array([vehicle['tx_m'], vehicle['ty_m'], vehicle['tz_m']])
                    
                    # Use vehicle position directly (already in ego frame)
                    vehicle_pos = vehicle_pos_annotation[:2]  # Only X, Y
                    
                    # Check if vehicle is within our BEV range
                    # AV2: +X=Forward, +Y=Left
                    av2_x_forward = vehicle_pos[0]
                    av2_y_left = vehicle_pos[1]
                    
                    # Check if vehicle is in front of ego (positive X in AV2)
                    if av2_x_forward < self.x_forward_lb or av2_x_forward > self.x_forward_ub:
                        continue
                    
                    # Check if vehicle is within left-right range
                    if av2_y_left < self.y_left_lb or av2_y_left > self.y_left_ub:
                        continue
                    
                    # Check if vehicle is visible in camera view
                    # Only include vehicles that can be seen in the front camera
                    vehicle_pos_3d = vehicle_pos_annotation.reshape(1, -1)
                    uv_cam = pinhole_camera.project_ego_to_img(vehicle_pos_3d)
                    
                    # uv_cam returns (coordinates, homogeneous_coords, is_valid_array)
                    if uv_cam is None or len(uv_cam) < 3:
                        continue
                    
                    coords, _, is_valid = uv_cam
                    
                    # Skip if not visible in camera (behind camera or outside FOV)
                    if len(is_valid) == 0 or not is_valid[0]:
                        continue
                    
                    # Check if vehicle is on road (skip if not on road)
                    if road_bev is not None:
                        # Convert ego coordinates to BEV image coordinates
                        bev_x = int((av2_y_left + self.range / 2) / self.res)
                        bev_y = int((self.range - av2_x_forward) / self.res)
                        
                        # Check if on road with small tolerance radius
                        on_road = False
                        radius = 3
                        for dy in range(-radius, radius+1):
                            for dx in range(-radius, radius+1):
                                check_x = bev_x + dx
                                check_y = bev_y + dy
                                if 0 <= check_x < self.occ_map_size and 0 <= check_y < self.occ_map_size:
                                    if road_bev[check_y, check_x] > 0:
                                        on_road = True
                                        break
                            if on_road:
                                break
                        
                        # Skip vehicles not on roads
                        if not on_road:
                            continue
                    
                    # Create a simple rectangular representation
                    # Use vehicle center and dimensions
                    length = vehicle['length_m']
                    width = vehicle['width_m']
                    
                    # Create corners of rectangle around vehicle center
                    # AV2: +X=Forward (length), +Y=Left (width)
                    half_length = length / 2
                    half_width = width / 2
                    
                    corners = np.array([
                        [vehicle_pos[0] - half_length, vehicle_pos[1] - half_width],  # Back-right
                        [vehicle_pos[0] + half_length, vehicle_pos[1] - half_width],  # Front-right
                        [vehicle_pos[0] + half_length, vehicle_pos[1] + half_width],  # Front-left
                        [vehicle_pos[0] - half_length, vehicle_pos[1] + half_width]   # Back-left
                    ])
                    
                    # Convert to BEV coordinates
                    bev_coords = self.world_to_bev(np.column_stack([corners, np.zeros(4)]))
                    
                    # Draw filled polygon
                    cv2.fillPoly(bev_image, [bev_coords], 255)
            
            # Save BEV image (use camera timestamp for filename to match road generation)
            output_path = os.path.join(vehicle_dir, f"{self.camera}_{camera_ts}.jpg")
            cv2.imwrite(output_path, bev_image)
    
    def generate_road_bev(self, log_id: str, output_dir: str):
        """Generate road BEV layouts for a single log."""
        print(f"Processing road layouts for log: {log_id}")
        
        # Create output directory
        road_dir = os.path.join(output_dir, log_id, "road_gt")
        os.makedirs(road_dir, exist_ok=True)
        
        # Load map
        map_path = os.path.join(self.args.base_path, self.args.split, log_id, "map")
        if not os.path.exists(map_path):
            print(f"Map directory not found: {map_path}")
            return
            
        static_map = ArgoverseStaticMap.from_map_dir(Path(map_path), build_raster=True)
        
        # Get ego poses to determine available timestamps
        ego_poses_path = os.path.join(self.args.base_path, self.args.split, log_id, "city_SE3_egovehicle.feather")
        if not os.path.exists(ego_poses_path):
            print(f"No ego poses found for {log_id}")
            return
            
        ego_poses_df = read_feather(Path(ego_poses_path))
        unique_timestamps = sorted(ego_poses_df['timestamp_ns'].unique())
        
        # Get available camera timestamps (same as vehicle generation)
        camera_dir = os.path.join(self.args.base_path, self.args.split, log_id, "sensors", "cameras", self.camera)
        if not os.path.exists(camera_dir):
            print(f"Camera directory not found: {camera_dir}")
            return
            
        camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.jpg')])
        camera_timestamps = [int(f.replace('.jpg', '')) for f in camera_files]
        
        def find_closest_ego_timestamp(target_timestamp):
            """Find the closest ego pose timestamp to the target camera timestamp."""
            if not unique_timestamps:
                return None
            closest_idx = min(range(len(unique_timestamps)), 
                             key=lambda i: abs(unique_timestamps[i] - target_timestamp))
            return unique_timestamps[closest_idx]
        
        for camera_timestamp in tqdm(camera_timestamps, desc=f"Processing {log_id} road"):
            # Find closest ego pose timestamp
            closest_ego_ts = find_closest_ego_timestamp(camera_timestamp)
            if closest_ego_ts is None:
                continue
                
            # Check if the timestamps are reasonably close (within 100ms)
            time_diff = abs(closest_ego_ts - camera_timestamp) / 1e6  # Convert to milliseconds
            if time_diff > 100:  # Skip if more than 100ms apart
                continue
            
            # Get ego pose using the closest ego timestamp
            ego_pose = self.dataloader.get_city_SE3_ego(log_id, closest_ego_ts)
            if ego_pose is None:
                continue
            
            # Create BEV image
            bev_image = np.zeros((self.occ_map_size, self.occ_map_size), dtype=np.uint8)
            
            # Get ego position
            ego_translation = ego_pose.translation
            ego_x, ego_y = ego_translation[0], ego_translation[1]
            
            # Get nearby lane segments within a reasonable radius
            search_radius = self.range * 1.5  # Search in 60m radius for 40m BEV
            
            # Get all lane segments and filter by distance
            all_lanes = static_map.get_scenario_lane_segments()
            nearby_lanes = []
            
            for lane in all_lanes:
                # Get lane center point (approximate)
                if lane.polygon_boundary is not None:
                    if hasattr(lane.polygon_boundary, 'exterior'):
                        lane_points = np.array(lane.polygon_boundary.exterior.coords)[:, :2]
                    elif isinstance(lane.polygon_boundary, np.ndarray):
                        lane_points = lane.polygon_boundary[:, :2]
                    else:
                        continue
                    
                    if len(lane_points) > 0:
                        # Check if any point of the lane is within search radius of ego
                        distances_to_ego = np.sqrt((lane_points[:, 0] - ego_x)**2 + (lane_points[:, 1] - ego_y)**2)
                        min_distance = np.min(distances_to_ego)
                        
                        if min_distance < search_radius:
                            nearby_lanes.append(lane)
            
            for lane_segment in nearby_lanes:
                # Get lane polygon
                lane_polygon = lane_segment.polygon_boundary
                if lane_polygon is None:
                    continue
                
                # Handle different polygon types
                if hasattr(lane_polygon, 'exterior'):
                    # Shapely polygon
                    lane_points_city = np.array(lane_polygon.exterior.coords)[:, :2]  # Get x, y only
                elif isinstance(lane_polygon, np.ndarray):
                    # Numpy array
                    lane_points_city = lane_polygon[:, :2]
                else:
                    # Try to convert to array
                    try:
                        lane_points_city = np.array(lane_polygon)[:, :2]
                    except:
                        continue
                
                # Transform from city to ego frame
                lane_points_ego = ego_pose.inverse().transform_point_cloud(
                    np.column_stack([lane_points_city, np.zeros(len(lane_points_city))])
                )[:, :2]  # Keep only x, y
                
                # Check if lane intersects with BEV bounds
                # AV2: +X=Forward [0, 40], +Y=Left [-20, 20]
                x_forward_overlaps = (lane_points_ego[:, 0].max() >= self.x_forward_lb) & (lane_points_ego[:, 0].min() <= self.x_forward_ub)
                y_left_overlaps = (lane_points_ego[:, 1].max() >= self.y_left_lb) & (lane_points_ego[:, 1].min() <= self.y_left_ub)
                
                if not (x_forward_overlaps and y_left_overlaps):
                    continue
                
                # For lanes that extend beyond BEV, create a clipped polygon
                # Use all points and let BEV coordinate conversion handle clipping
                valid_points = lane_points_ego
                
                # Convert to BEV coordinates
                bev_coords = self.world_to_bev(valid_points)
                
                # Draw lane
                if len(bev_coords) > 2:
                    cv2.fillPoly(bev_image, [bev_coords], 255)
            
            # Save BEV image (use camera timestamp for filename to match vehicle generation)
            output_path = os.path.join(road_dir, f"{self.camera}_{camera_timestamp}.png")
            cv2.imwrite(output_path, bev_image)
    
    def process_dataset(self):
        """Process the entire dataset."""
        base_dir = os.path.join(self.args.base_path, self.args.split)
        out_dir = base_dir if self.args.out_dir == "" else self.args.out_dir
        
        # Get all log IDs
        log_ids = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
        
        if self.args.max_logs:
            log_ids = log_ids[:self.args.max_logs]
            
        print(f"Processing {len(log_ids)} logs for {self.args.seg_class} generation")
        
        for log_id in log_ids:
            try:
                output_dir = os.path.join(out_dir, log_id)
                os.makedirs(output_dir, exist_ok=True)
                
                if self.args.seg_class == "vehicle":
                    self.generate_vehicle_bev(log_id, out_dir)
                elif self.args.seg_class == "road":
                    self.generate_road_bev(log_id, out_dir)
                    
            except Exception as e:
                print(f"Error processing log {log_id}: {e}")
                continue
        
        print(f"Completed processing {len(log_ids)} logs")


def main():
    args = get_args()
    
    # Validate paths
    if not os.path.exists(args.base_path):
        print(f"Error: Base path does not exist: {args.base_path}")
        sys.exit(1)
    
    split_path = os.path.join(args.base_path, args.split)
    if not os.path.exists(split_path):
        print(f"Error: Split path does not exist: {split_path}")
        sys.exit(1)
    
    # Initialize generator and process
    generator = ArgoverseV2LayoutGenerator(args)
    generator.process_dataset()


if __name__ == "__main__":
    main()
