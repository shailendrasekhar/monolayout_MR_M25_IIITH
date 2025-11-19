#!/usr/bin/env python3
"""
Detailed check of vehicle-road alignment with individual vehicle analysis
"""

import cv2
import numpy as np
import os

def check_detailed_alignment():
    """Check alignment for each individual vehicle."""
    
    base_path = '/ssd_scratch/cvit/varunp/argoverse/train'
    log_id = '1842383a-1577-3b7a-90db-41a9a6668ee2'
    
    # Load existing generated images
    road_dir = os.path.join(base_path, log_id, 'road_gt')
    vehicle_dir = os.path.join(base_path, log_id, 'car_bev_gt')
    
    # Find a timestamp with both road and vehicle content
    road_files = sorted(os.listdir(road_dir))
    
    for road_file in road_files[:10]:
        timestamp = road_file.replace('ring_front_center_', '').replace('.png', '')
        vehicle_file = f"ring_front_center_{timestamp}.jpg"
        
        road_path = os.path.join(road_dir, road_file)
        vehicle_path = os.path.join(vehicle_dir, vehicle_file)
        
        if not os.path.exists(vehicle_path):
            continue
            
        road_img = cv2.imread(road_path, cv2.IMREAD_GRAYSCALE)
        vehicle_img = cv2.imread(vehicle_path, cv2.IMREAD_GRAYSCALE)
        
        if road_img is None or vehicle_img is None:
            continue
            
        # Check if both have content
        road_content = len(np.unique(road_img)) > 1
        vehicle_content = len(np.unique(vehicle_img)) > 1
        
        if road_content and vehicle_content:
            print(f"\n=== Analyzing timestamp: {timestamp} ===")
            
            # Create overlay analysis
            h, w = road_img.shape
            
            # Create colored overlay
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            overlay[:, :, 1] = road_img  # Green for roads
            overlay[:, :, 2] = vehicle_img  # Red for vehicles
            
            # Calculate alignment metrics
            road_pixels = np.sum(road_img > 0)
            vehicle_pixels = np.sum(vehicle_img > 0)
            overlap_pixels = np.sum((road_img > 0) & (vehicle_img > 0))
            
            print(f"Road pixels: {road_pixels}")
            print(f"Vehicle pixels: {vehicle_pixels}")
            print(f"Overlap pixels: {overlap_pixels}")
            
            if vehicle_pixels > 0:
                overlap_ratio = overlap_pixels / vehicle_pixels
                print(f"Vehicle-road overlap: {overlap_ratio:.2%}")
                
                # Analyze vehicle distribution
                vehicle_coords = np.where(vehicle_img > 0)
                if len(vehicle_coords[0]) > 0:
                    # Find vehicle center of mass
                    vehicle_center_y = np.mean(vehicle_coords[0])
                    vehicle_center_x = np.mean(vehicle_coords[1])
                    
                    # Find road center of mass
                    road_coords = np.where(road_img > 0)
                    if len(road_coords[0]) > 0:
                        road_center_y = np.mean(road_coords[0])
                        road_center_x = np.mean(road_coords[1])
                        
                        # Calculate center distance
                        center_distance = np.sqrt((vehicle_center_x - road_center_x)**2 + 
                                                (vehicle_center_y - road_center_y)**2)
                        
                        print(f"Vehicle center: ({vehicle_center_x:.1f}, {vehicle_center_y:.1f})")
                        print(f"Road center: ({road_center_x:.1f}, {road_center_y:.1f})")
                        print(f"Center distance: {center_distance:.1f} pixels")
                        
                        # Add center markers to overlay
                        cv2.circle(overlay, (int(vehicle_center_x), int(vehicle_center_y)), 5, (255, 255, 255), -1)  # White for vehicle center
                        cv2.circle(overlay, (int(road_center_x), int(road_center_y)), 5, (0, 255, 255), -1)  # Cyan for road center
                        cv2.line(overlay, (int(vehicle_center_x), int(vehicle_center_y)), 
                                (int(road_center_x), int(road_center_y)), (255, 255, 0), 2)  # Yellow line
            
            # Add coordinate grid for reference
            grid_spacing = 32
            for i in range(0, w, grid_spacing):
                cv2.line(overlay, (i, 0), (i, h), (64, 64, 64), 1)
            for i in range(0, h, grid_spacing):
                cv2.line(overlay, (0, i), (w, i), (64, 64, 64), 1)
            
            # Add coordinate labels
            cv2.putText(overlay, 'Y=0 (40m ahead)', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay, f'Y={h-1} (ego)', (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay, 'X=0 (-20m left)', (5, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay, f'X={w-1} (+20m right)', (w-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save detailed analysis
            output_path = f"detailed_alignment_{timestamp}.png"
            cv2.imwrite(output_path, overlay)
            print(f"Saved: {output_path}")
            print("Green=Roads, Red=Vehicles, Yellow=Overlap, White=Vehicle center, Cyan=Road center")
            
            # Only analyze first valid timestamp
            break
    
    print("\nDetailed alignment analysis complete!")
    print("Check the saved image to see exact vehicle-road positioning")


if __name__ == "__main__":
    check_detailed_alignment()
