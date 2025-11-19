#!/usr/bin/env python3
"""
Investigate whether vehicle annotations are in ego frame or city frame
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.append('/home2/varun.paturkar/miniconda3/envs/monolayout/lib/python3.10/site-packages')

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.utils.io import read_feather


def investigate_coordinate_frames():
    """Check if vehicle annotations are in ego or city frame."""
    
    base_path = "/ssd_scratch/cvit/varunp/argoverse"
    split = "train"
    log_id = "1842383a-1577-3b7a-90db-41a9a6668ee2"
    
    # Initialize dataloader
    dataloader = AV2SensorDataLoader(
        data_dir=Path(base_path) / split,
        labels_dir=Path(base_path) / split
    )
    
    # Get annotations
    annotations_path = os.path.join(base_path, split, log_id, "annotations.feather")
    annotations_df = read_feather(Path(annotations_path))
    
    # Get a timestamp with vehicles
    unique_timestamps = sorted(annotations_df['timestamp_ns'].unique())
    test_timestamp = unique_timestamps[10]
    
    frame_annotations = annotations_df[annotations_df['timestamp_ns'] == test_timestamp]
    vehicle_categories = ['REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER']
    vehicles = frame_annotations[frame_annotations['category'].isin(vehicle_categories)]
    
    if len(vehicles) == 0:
        print("No vehicles found")
        return
    
    # Get ego pose
    ego_pose = dataloader.get_city_SE3_ego(log_id, test_timestamp)
    ego_city = ego_pose.translation
    
    print(f"=== Coordinate Frame Investigation ===")
    print(f"Timestamp: {test_timestamp}")
    print(f"Ego position in city frame: ({ego_city[0]:.2f}, {ego_city[1]:.2f})")
    print(f"Number of vehicles: {len(vehicles)}")
    
    print(f"\n=== Vehicle Analysis ===")
    for i, (_, vehicle) in enumerate(vehicles.head(5).iterrows()):
        vehicle_pos = np.array([vehicle['tx_m'], vehicle['ty_m'], vehicle['tz_m']])
        
        print(f"\nVehicle {i+1}:")
        print(f"  Annotation coords: ({vehicle_pos[0]:.2f}, {vehicle_pos[1]:.2f})")
        
        # Test 1: If annotations are in city frame
        distance_from_ego_city = np.sqrt((vehicle_pos[0] - ego_city[0])**2 + (vehicle_pos[1] - ego_city[1])**2)
        print(f"  Distance from ego (if city frame): {distance_from_ego_city:.2f}m")
        
        # Test 2: If annotations are in ego frame (should be small distances)
        distance_from_origin = np.sqrt(vehicle_pos[0]**2 + vehicle_pos[1]**2)
        print(f"  Distance from origin (if ego frame): {distance_from_origin:.2f}m")
        
        # Test 3: Transform annotation from city to ego frame
        if distance_from_ego_city < 200:  # Only if reasonably close
            vehicle_city_3d = np.array([[vehicle_pos[0], vehicle_pos[1], vehicle_pos[2]]])
            vehicle_ego_transformed = ego_pose.inverse().transform_point_cloud(vehicle_city_3d)[0]
            print(f"  If city->ego transform: ({vehicle_ego_transformed[0]:.2f}, {vehicle_ego_transformed[1]:.2f})")
            
            distance_transformed = np.sqrt(vehicle_ego_transformed[0]**2 + vehicle_ego_transformed[1]**2)
            print(f"  Distance after transform: {distance_transformed:.2f}m")
    
    # Determine which interpretation makes more sense
    print(f"\n=== Analysis ===")
    
    # Check typical distances
    vehicle_positions = []
    for _, vehicle in vehicles.iterrows():
        vehicle_pos = np.array([vehicle['tx_m'], vehicle['ty_m']])
        vehicle_positions.append(vehicle_pos)
    
    vehicle_positions = np.array(vehicle_positions)
    
    # Distance from ego city position
    distances_from_ego_city = np.sqrt(np.sum((vehicle_positions - ego_city[:2])**2, axis=1))
    avg_distance_city = np.mean(distances_from_ego_city)
    
    # Distance from origin (ego frame assumption)
    distances_from_origin = np.sqrt(np.sum(vehicle_positions**2, axis=1))
    avg_distance_origin = np.mean(distances_from_origin)
    
    print(f"Average distance from ego city position: {avg_distance_city:.2f}m")
    print(f"Average distance from origin: {avg_distance_origin:.2f}m")
    
    if avg_distance_origin < 100 and avg_distance_city > 1000:
        print("✅ Vehicle annotations appear to be in EGO FRAME")
        return "ego"
    elif avg_distance_city < 100 and avg_distance_origin > 1000:
        print("✅ Vehicle annotations appear to be in CITY FRAME")
        return "city"
    else:
        print("❓ Unclear - need manual inspection")
        return "unclear"


if __name__ == "__main__":
    investigate_coordinate_frames()
