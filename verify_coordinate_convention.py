#!/usr/bin/env python3
"""
Verify the actual coordinate system convention used in Argoverse 2
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.append('/home2/varun.paturkar/miniconda3/envs/monolayout/lib/python3.10/site-packages')

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.utils.io import read_feather


def verify_coordinate_convention():
    """Verify the coordinate system convention."""
    
    print("🧭 VERIFYING COORDINATE SYSTEM CONVENTION")
    print("="*50)
    
    # Setup
    base_path = "/ssd_scratch/cvit/varunp/argoverse"
    split = "train"
    log_id = "1842383a-1577-3b7a-90db-41a9a6668ee2"
    
    dataloader = AV2SensorDataLoader(
        data_dir=Path(base_path) / split,
        labels_dir=Path(base_path) / split
    )
    
    # Get multiple timestamps to analyze trajectory
    annotations_path = os.path.join(base_path, split, log_id, "annotations.feather")
    annotations_df = read_feather(Path(annotations_path))
    unique_timestamps = sorted(annotations_df['timestamp_ns'].unique())
    
    print(f"Analyzing trajectory over {len(unique_timestamps)} timestamps...")
    
    # Get ego positions over time
    positions = []
    timestamps_used = []
    
    for i in range(0, min(50, len(unique_timestamps)), 5):  # Every 5th timestamp, up to 50
        timestamp = unique_timestamps[i]
        ego_pose = dataloader.get_city_SE3_ego(log_id, timestamp)
        if ego_pose is not None:
            pos = ego_pose.translation
            positions.append([pos[0], pos[1]])
            timestamps_used.append(timestamp)
    
    positions = np.array(positions)
    
    print(f"Collected {len(positions)} ego positions")
    print(f"Position range: X=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], Y=[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")
    
    # Calculate overall trajectory direction in city frame
    start_pos = positions[0]
    end_pos = positions[-1]
    overall_movement = end_pos - start_pos
    overall_distance = np.linalg.norm(overall_movement)
    
    print(f"\nOverall trajectory (city frame):")
    print(f"  Start: [{start_pos[0]:.2f}, {start_pos[1]:.2f}]")
    print(f"  End:   [{end_pos[0]:.2f}, {end_pos[1]:.2f}]")
    print(f"  Movement: [{overall_movement[0]:.2f}, {overall_movement[1]:.2f}]")
    print(f"  Distance: {overall_distance:.2f}m")
    
    # Analyze movement in ego frame for multiple segments
    print(f"\nMovement analysis in ego frame:")
    
    movements_ego = []
    for i in range(len(timestamps_used) - 1):
        curr_timestamp = timestamps_used[i+1]
        prev_timestamp = timestamps_used[i]
        
        curr_pose = dataloader.get_city_SE3_ego(log_id, curr_timestamp)
        prev_pose = dataloader.get_city_SE3_ego(log_id, prev_timestamp)
        
        if curr_pose is not None and prev_pose is not None:
            # Transform previous position to current ego frame
            prev_pos_3d = np.array([[prev_pose.translation[0], prev_pose.translation[1], prev_pose.translation[2]]])
            prev_in_curr_ego = curr_pose.inverse().transform_point_cloud(prev_pos_3d)[0]
            
            # Movement in ego frame (from prev to curr, where curr is at origin)
            movement_ego = -prev_in_curr_ego[:2]  # Negative because we moved FROM prev TO curr
            movements_ego.append(movement_ego)
            
            if len(movements_ego) <= 5:  # Print first 5
                print(f"  Segment {len(movements_ego)}: [{movement_ego[0]:6.2f}, {movement_ego[1]:6.2f}]")
    
    movements_ego = np.array(movements_ego)
    
    # Analyze predominant direction
    avg_movement = np.mean(movements_ego, axis=0)
    total_x_movement = np.sum(movements_ego[:, 0])
    total_y_movement = np.sum(movements_ego[:, 1])
    
    print(f"\nMovement statistics:")
    print(f"  Average movement per segment: [{avg_movement[0]:.2f}, {avg_movement[1]:.2f}]")
    print(f"  Total X movement: {total_x_movement:.2f}m")
    print(f"  Total Y movement: {total_y_movement:.2f}m")
    
    # Determine coordinate convention
    print(f"\n🧭 COORDINATE CONVENTION ANALYSIS:")
    
    if abs(total_x_movement) > abs(total_y_movement):
        primary_axis = "X"
        primary_movement = total_x_movement
    else:
        primary_axis = "Y" 
        primary_movement = total_y_movement
    
    print(f"  Primary movement axis: {primary_axis}")
    print(f"  Primary movement magnitude: {primary_movement:.2f}m")
    
    if primary_axis == "X":
        if primary_movement > 0:
            print(f"  ✅ +X appears to be the FORWARD direction")
            print(f"  📝 Coordinate convention: +X=Forward, +Y=Left, +Z=Up")
        else:
            print(f"  ✅ -X appears to be the FORWARD direction")
            print(f"  📝 Coordinate convention: -X=Forward, +Y=Right, +Z=Up")
    else:
        if primary_movement > 0:
            print(f"  ✅ +Y appears to be the FORWARD direction")
            print(f"  📝 Coordinate convention: +Y=Forward, +X=Right, +Z=Up")
        else:
            print(f"  ✅ -Y appears to be the FORWARD direction")
            print(f"  📝 Coordinate convention: -Y=Forward, +X=Left, +Z=Up")
    
    # Test with a known forward point
    print(f"\n🔍 VERIFICATION TEST:")
    test_timestamp = timestamps_used[len(timestamps_used)//2]  # Middle timestamp
    test_pose = dataloader.get_city_SE3_ego(log_id, test_timestamp)
    
    # Test what "10m forward" means in different conventions
    test_points = {
        "+X forward": np.array([[10, 0, 0]]),
        "+Y forward": np.array([[0, 10, 0]]),
        "-X forward": np.array([[-10, 0, 0]]),
        "-Y forward": np.array([[0, -10, 0]])
    }
    
    print(f"  Testing 10m in different directions (ego -> city):")
    for direction, point_ego in test_points.items():
        point_city = test_pose.transform_point_cloud(point_ego)[0]
        city_movement = point_city[:2] - test_pose.translation[:2]
        print(f"    {direction:12}: city movement [{city_movement[0]:6.2f}, {city_movement[1]:6.2f}]")
    
    return primary_axis, primary_movement


if __name__ == "__main__":
    verify_coordinate_convention()
