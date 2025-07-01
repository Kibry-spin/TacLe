#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
Simple and efficient dataset visualizer that only loads specified episode data.

This script is designed to be memory and disk efficient by:
1. Only loading the specific episode data needed
2. Loading episodes one frame at a time
3. Avoiding loading the entire dataset into memory

Enhanced Tactile Sensor Support:
- Tac3D sensors: 3D tactile visualization with force/displacement vectors
- GelSight sensors: RGB tactile images and statistical analysis

Examples:

- Visualize specific episode efficiently:
```
python lerobot/scripts/visualize_dataset_simple.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize with tactile sensors:
```
python lerobot/scripts/visualize_dataset_simple.py \
    --repo-id your_dataset_with_tactile \
    --episode-index 0
    python lerobot/scripts/visualize_dataset_simple.py  --repo-id F:\Two_Sensor_dataset\Real_test3  --episode-index 0
```
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 torch tensor to HWC uint8 numpy array."""
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def generate_tac3d_mesh_connectivity(nx: int = 20, ny: int = 20):
    """Generate Tac3D sensor mesh connectivity information."""
    connect = []
    for iy in range(ny-1):
        for ix in range(nx-1):
            idx = iy * nx + ix
            # Generate two triangles per grid cell
            connect.append([idx, idx+1, idx+nx])
            connect.append([idx+nx+1, idx+nx, idx+1])
    return np.array(connect)


def generate_tac3d_demo_data():
    """Generate Tac3D demo data (used when actual data is all zeros)."""
    # Generate 20x20 grid positions
    nx, ny = 20, 20
    x = np.linspace(-8, 8, nx)
    y = np.linspace(-8, 8, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Generate radial force field (larger forces in center region)
    forces = np.zeros((400, 3))
    center_x, center_y = 10, 10
    
    for i in range(20):
        for j in range(20):
            idx = i * 20 + j
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            
            if dist < 8:
                force_magnitude = 1.0 * np.exp(-dist/3)
                if dist > 0:
                    fx = force_magnitude * (i - center_x) / dist
                    fy = force_magnitude * (j - center_y) / dist
                else:
                    fx = fy = 0
                fz = force_magnitude * 0.5
                forces[idx] = [fx, fy, fz]
    
    # Generate displacement data (proportional to forces)
    displacements = forces * 0.1
    
    return positions, displacements, forces


def visualize_tac3d_points_and_vectors(
    sensor_name: str, 
    positions: np.ndarray, 
    displacements: np.ndarray | None = None, 
    forces: np.ndarray | None = None,
    scale_displacement: float = 5.0,
    scale_force: float = 30.0
):
    """Visualize Tac3D sensor 3D point cloud and vector fields."""
    if positions is None or positions.size == 0:
        return
        
    # Ensure correct data format
    if positions.ndim != 2 or positions.shape[1] != 3:
        print(f"Warning: Invalid positions shape: {positions.shape}, expected (N, 3)")
        return
    
    # Check if data is all zeros (sensor not properly initialized)
    positions_zero = np.all(positions == 0)
    forces_zero = forces is None or np.all(forces == 0) if forces is not None else True
    displacements_zero = displacements is None or np.all(displacements == 0) if displacements is not None else True
    
    if positions_zero and forces_zero and displacements_zero:
        # If all data is zero, use demo data and show warning
        print(f"⚠️  Warning: Tac3D sensor {sensor_name} data is all zeros. Using demo data for visualization.")
        
        demo_positions, demo_displacements, demo_forces = generate_tac3d_demo_data()
        
        # Log warning
        rr.log(f"tactile/tac3d/{sensor_name}/sensor_warning", 
               rr.TextLog("⚠️ DEMO DATA: Real sensor data is all zeros!\n"
                         "This indicates sensor initialization or calibration issues.\n"
                         "Showing demo data to demonstrate visualization capabilities."))
        
        # Use demo data for visualization
        positions = demo_positions
        displacements = demo_displacements  
        forces = demo_forces
        
        # Reduce scale factors for demo data
        scale_force = scale_force * 0.5
        scale_displacement = scale_displacement * 0.5
    
    try:
        # 1. Visualize 3D tactile sensor surface mesh
        if positions.shape[0] == 400:  # Standard 20x20 grid
            # Generate mesh connectivity
            mesh_triangles = generate_tac3d_mesh_connectivity(20, 20)
            
            # Color mapping based on Z coordinates
            z_min, z_max = float(positions[:, 2].min()), float(positions[:, 2].max())
            if z_max > z_min:
                z_normalized = (positions[:, 2] - z_min) / (z_max - z_min)
                # Blue to light purple gradient
                vertex_colors = np.zeros((len(positions), 3))
                vertex_colors[:, 0] = 0.4 + 0.4 * z_normalized  # Red component
                vertex_colors[:, 1] = 0.4 + 0.4 * z_normalized  # Green component  
                vertex_colors[:, 2] = 0.7 + 0.3 * z_normalized  # Blue component
            else:
                # Uniform light purple color
                vertex_colors = np.array([[0.6, 0.6, 0.9]] * len(positions))
            
            # Log 3D mesh surface
            rr.log(f"tactile/tac3d/{sensor_name}/surface_mesh", 
                   rr.Mesh3D(
                       vertex_positions=positions,
                       triangle_indices=mesh_triangles,
                       vertex_colors=vertex_colors
                   ))
        
        # 2. Visualize 3D tactile sensing points (as point cloud)
        rr.log(f"tactile/tac3d/{sensor_name}/sensing_points", 
               rr.Points3D(positions, colors=vertex_colors if 'vertex_colors' in locals() else None, radii=0.15))
        
        # 3. Visualize displacement vector field
        if displacements is not None and displacements.size > 0:
            if displacements.shape == positions.shape:
                # Calculate displacement magnitudes
                displacement_magnitudes = np.linalg.norm(displacements, axis=1)
                max_displacement = displacement_magnitudes.max()
                
                if max_displacement > 0:
                    # Filter significant displacements (avoid noise)
                    threshold = max_displacement * 0.05  # 5% threshold
                    significant_mask = displacement_magnitudes > threshold
                    
                    if np.any(significant_mask):
                        start_points = positions[significant_mask]
                        # Apply scale factor
                        displacement_vectors = displacements[significant_mask] * scale_displacement
                        
                        # Green color coding (displacement magnitude)
                        disp_normalized = displacement_magnitudes[significant_mask] / max_displacement
                        vector_colors = np.zeros((len(start_points), 3))
                        vector_colors[:, 1] = 0.3 + 0.7 * disp_normalized  # Green main tone
                        vector_colors[:, 0] = 0.2 * (1 - disp_normalized)  # Little red contrast
                        vector_colors[:, 2] = 0.1  # Little blue
                        
                        # Log displacement arrows
                        rr.log(f"tactile/tac3d/{sensor_name}/displacement_arrows",
                               rr.Arrows3D(
                                   origins=start_points, 
                                   vectors=displacement_vectors, 
                                   colors=vector_colors,
                                   radii=0.05
                               ))
                        
                        # Displacement statistics
                        rr.log(f"tactile/tac3d/{sensor_name}/displacement_stats", 
                               rr.TextLog(f"Max displacement: {max_displacement:.4f}mm, "
                                         f"Mean: {np.mean(displacement_magnitudes):.4f}mm, "
                                         f"Active points: {np.sum(significant_mask)}/400"))
        
        # 4. Visualize force vector field for all 400 points
        if forces is not None and forces.size > 0:
            if forces.shape == positions.shape and forces.shape[0] == 400:
                # Calculate force magnitudes
                force_magnitudes = np.linalg.norm(forces, axis=1)
                max_force = force_magnitudes.max()
                mean_force = np.mean(force_magnitudes)
                
                # Show force vectors for all 400 points, no filtering
                # Apply scale factor
                force_vectors = forces * scale_force
                
                # Color coding based on force magnitude (red gradient)
                if max_force > 0:
                    force_normalized = force_magnitudes / max_force
                else:
                    force_normalized = np.zeros(400)
                
                # Create color mapping for 20x20 grid
                force_colors = np.zeros((400, 3))
                for i in range(400):
                    intensity = force_normalized[i]
                    # Red main tone, higher intensity = more red
                    force_colors[i, 0] = 0.3 + 0.7 * intensity  # Red component [0.3, 1.0]
                    force_colors[i, 1] = 0.1 * (1 - intensity)  # Green component [0.0, 0.1]
                    force_colors[i, 2] = 0.1 * (1 - intensity)  # Blue component [0.0, 0.1]
                
                # Log force arrows for all 400 points
                rr.log(f"tactile/tac3d/{sensor_name}/force_arrows_all",
                       rr.Arrows3D(
                           origins=positions,  # All 400 points as origins
                           vectors=force_vectors,  # All 400 force vectors
                           colors=force_colors,  # Color for each point
                           radii=0.05
                       ))
                
                # Additionally show significant force points (with thicker arrows)
                if max_force > 0:
                    threshold = max_force * 0.1  # 10% threshold
                    significant_mask = force_magnitudes > threshold
                    
                    if np.any(significant_mask):
                        significant_positions = positions[significant_mask]
                        significant_forces = force_vectors[significant_mask]
                        significant_colors = force_colors[significant_mask]
                        
                        # Bright colors for significant force points
                        bright_colors = np.copy(significant_colors)
                        bright_colors[:, 0] = np.minimum(bright_colors[:, 0] * 1.5, 1.0)  # Enhance red
                        
                        rr.log(f"tactile/tac3d/{sensor_name}/force_arrows_significant",
                               rr.Arrows3D(
                                   origins=significant_positions,
                                   vectors=significant_forces,
                                   colors=bright_colors,
                                   radii=0.12  # Thicker arrows
                               ))
                
                # Force statistics and grid info
                active_points = np.sum(force_magnitudes > mean_force * 0.1)
                rr.log(f"tactile/tac3d/{sensor_name}/force_grid_stats", 
                       rr.TextLog(f"20x20 Force Grid Visualization:\n"
                                 f"Max force: {max_force:.4f}N\n"
                                 f"Mean force: {mean_force:.4f}N\n"
                                 f"Active points (>10% mean): {active_points}/400\n"
                                 f"Force scale factor: {scale_force}x"))
        
        # 5. Visualize sensor bounding box
        if positions.size > 0:
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            center = (min_pos + max_pos) / 2
            size = max_pos - min_pos
            
            # Semi-transparent bounding box
            box_corners = np.array([
                [min_pos[0], min_pos[1], min_pos[2]],  # Bottom four corners
                [max_pos[0], min_pos[1], min_pos[2]],
                [max_pos[0], max_pos[1], min_pos[2]],
                [min_pos[0], max_pos[1], min_pos[2]],
                [min_pos[0], min_pos[1], max_pos[2]],  # Top four corners
                [max_pos[0], min_pos[1], max_pos[2]],
                [max_pos[0], max_pos[1], max_pos[2]],
                [min_pos[0], max_pos[1], max_pos[2]],
            ])
            
            # Bounding box wireframe
            box_lines = np.array([
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7],  # Connecting lines
            ])
            
            rr.log(f"tactile/tac3d/{sensor_name}/sensor_bounds",
                   rr.LineStrips3D([box_corners[box_lines.flatten()]], colors=[0.3, 0.3, 0.3]))
            
            # Coordinate axes
            axis_length = np.max(size) * 0.3
            axes_origins = np.array([[center[0], center[1], min_pos[2]]] * 3)
            axes_vectors = np.array([
                [axis_length, 0, 0],  # X-axis - red
                [0, axis_length, 0],  # Y-axis - green  
                [0, 0, axis_length],  # Z-axis - blue
            ])
            axes_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            
            rr.log(f"tactile/tac3d/{sensor_name}/coordinate_axes",
                   rr.Arrows3D(origins=axes_origins, vectors=axes_vectors, colors=axes_colors))
            
            # Sensor dimension info
            rr.log(f"tactile/tac3d/{sensor_name}/sensor_info", 
                   rr.TextLog(f"Sensor dimensions: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} mm\n"
                             f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) mm\n"
                             f"Total sensing points: {len(positions)}"))
                             
    except Exception as e:
        print(f"Warning: Error visualizing Tac3D data for {sensor_name}: {e}")


def get_episode_frame_indices(dataset: LeRobotDataset, episode_index: int):
    """Get the frame indices for a specific episode."""
    # When loading a dataset with episodes=[episode_index], the episode_data_index
    # will only contain one episode at index 0, regardless of the original episode_index
    if len(dataset.episode_data_index["from"]) == 1:
        # Only one episode loaded, use index 0
        from_idx = int(dataset.episode_data_index["from"][0].item())
        to_idx = int(dataset.episode_data_index["to"][0].item())
    else:
        # Multiple episodes loaded, use the original episode_index
        from_idx = int(dataset.episode_data_index["from"][episode_index].item())
        to_idx = int(dataset.episode_data_index["to"][episode_index].item())
    return range(from_idx, to_idx)


def visualize_dataset_simple(
    dataset: LeRobotDataset,
    episode_index: int,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    """
    Efficiently visualize a single episode by loading frames one at a time.
    """
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init`
    gc.collect()

    if mode == "distant":
        print(f"Starting server on web_port={web_port}, ws_port={ws_port}")
        print(f"Please use: rerun ws://localhost:{ws_port} from your local machine")

    logging.info("Getting episode frame indices")
    frame_indices = get_episode_frame_indices(dataset, episode_index)
    
    logging.info(f"Logging {len(frame_indices)} frames to Rerun")

    # Process frames one at a time to save memory
    for frame_idx in tqdm.tqdm(frame_indices, desc="Processing frames"):
        # Load single frame
        frame_data = dataset[frame_idx]
        
        # Set time information
        rr.set_time("frame_index", sequence=frame_data["frame_index"].item())
        rr.set_time("timestamp", timestamp=frame_data["timestamp"].item())

        # Display each camera image
        for key in dataset.meta.camera_keys:
            rr.log(key, rr.Image(to_hwc_uint8_numpy(frame_data[key])))

        # Display each dimension of action space
        if "action" in frame_data:
            for dim_idx, val in enumerate(frame_data["action"]):
                rr.log(f"action/{dim_idx}", rr.Scalars(val.item()))

        # Display each dimension of observed state space
        if "observation.state" in frame_data:
            for dim_idx, val in enumerate(frame_data["observation.state"]):
                rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

        if "next.done" in frame_data:
            rr.log("next.done", rr.Scalars(frame_data["next.done"].item()))

        if "next.reward" in frame_data:
            rr.log("next.reward", rr.Scalars(frame_data["next.reward"].item()))

        if "next.success" in frame_data:
            rr.log("next.success", rr.Scalars(frame_data["next.success"].item()))

        # Enhanced tactile sensor data visualization
        tac3d_sensors = {}  # {sensor_name: {data_type: data}}
        gelsight_sensors = {}  # {sensor_name: {data_type: data}}
        
        for key in frame_data.keys():
            if key.startswith("observation.tactile."):
                # Support two data structures:
                # New structure: observation.tactile.{sensor_type}.{name}.{field}
                # Old structure: observation.tactile.{name}.{field}
                parts = key.split(".")
                
                if len(parts) >= 5:
                    # New hierarchical structure: observation.tactile.{sensor_type}.{name}.{field}
                    sensor_type = parts[2]  # "gelsight" or "tac3d"
                    sensor_name = parts[3]  # "main_gripper0", "main_gripper1", etc.
                    data_type = parts[4]    # "tactile_image", "resultant_force", etc.
                elif len(parts) >= 4:
                    # Old flat structure: observation.tactile.{name}.{field}
                    sensor_name = parts[2]  # "left_gripper", "right_gripper", etc.
                    data_type = parts[3]    # "tactile_image", "resultant_force", etc.
                    
                    # Infer sensor type from data type
                    if data_type in ["tactile_image"]:
                        sensor_type = "gelsight"
                    elif data_type in ["resultant_force", "resultant_moment", "positions_3d", "forces_3d", "displacements_3d", "sensor_sn", "frame_index", "send_timestamp", "recv_timestamp"]:
                        sensor_type = "tac3d"
                    else:
                        sensor_type = "unknown"
                else:
                    continue  # Skip keys that don't match format

                # Get current data
                current_data = frame_data[key]
                if isinstance(current_data, torch.Tensor):
                    current_data = current_data.numpy()

                # Process by sensor type
                if sensor_type == "tac3d":
                    # Tac3D sensor data processing
                    if sensor_name not in tac3d_sensors:
                        tac3d_sensors[sensor_name] = {}
                    
                    # Field name mapping (数据集键名 -> Tac3D原始键名)
                    field_mapping = {
                        'positions_3d': '3D_Positions',
                        'displacements_3d': '3D_Displacements',
                        'forces_3d': '3D_Forces',
                        'resultant_force': 'resultant_force',  # 使用tac3d.py中添加的标准化字段名
                        'resultant_moment': 'resultant_moment',  # 使用tac3d.py中添加的标准化字段名
                        'sensor_sn': 'SN',
                        'frame_index': 'index',
                        'send_timestamp': 'sendTimestamp',
                        'recv_timestamp': 'recvTimestamp'
                    }
                    
                    # Convert field names
                    data_type_mapped = field_mapping.get(data_type, data_type)
                    tac3d_sensors[sensor_name][data_type_mapped] = current_data
                    
                    # Process force and moment data
                    if data_type == 'resultant_force' or data_type_mapped == 'resultant_force' or data_type_mapped == '3D_ResultantForce':
                        force = current_data
                        if force is None:
                            print(f"警告: 传感器 {sensor_name} 的合力数据为空")
                            continue
                            
                        # 打印调试信息
                        print(f"合力数据 ({sensor_name}): 类型={type(force)}, 形状={force.shape if hasattr(force, 'shape') else '无形状'}, 值={force}")
                        
                        # 处理不同形状的数据
                        if isinstance(force, np.ndarray):
                            if force.ndim == 2 and force.shape[0] == 1:
                                force = force[0]  # 从(1,3)转为(3,)
                            
                            if force.size >= 3:
                                # 记录xyz力分量
                                for dim_idx, force_component in enumerate(force[:3]):
                                    axis_name = ['x', 'y', 'z'][dim_idx]
                                    rr.log(f"Tactile/Tac3D/{sensor_name}/Forces/Component/{axis_name}", 
                                          rr.Scalars(force_component))
                                
                                # 记录合力大小
                                force_magnitude = np.linalg.norm(force[:3])
                                rr.log(f"Tactile/Tac3D/{sensor_name}/Forces/Magnitude", 
                                      rr.Scalars(force_magnitude))
                                
                                # 打印调试信息
                                print(f"合力分量 ({sensor_name}): X={force[0]:.4f}, Y={force[1]:.4f}, Z={force[2]:.4f}, 大小={force_magnitude:.4f}")
                            
                    elif data_type == 'resultant_moment' or data_type_mapped == 'resultant_moment' or data_type_mapped == '3D_ResultantMoment':
                        moment = current_data
                        if moment is None:
                            continue
                            
                        # 处理不同形状的数据
                        if isinstance(moment, np.ndarray):
                            if moment.ndim == 2 and moment.shape[0] == 1:
                                moment = moment[0]  # 从(1,3)转为(3,)
                            
                            if moment.size >= 3:
                                # 记录xyz力矩分量
                                for dim_idx, moment_component in enumerate(moment[:3]):
                                    axis_name = ['x', 'y', 'z'][dim_idx]
                                    rr.log(f"Tactile/Tac3D/{sensor_name}/Moments/Component/{axis_name}", 
                                          rr.Scalars(moment_component))
                                
                                # 记录力矩大小
                                moment_magnitude = np.linalg.norm(moment[:3])
                                rr.log(f"Tactile/Tac3D/{sensor_name}/Moments/Magnitude", 
                                      rr.Scalars(moment_magnitude))

                elif sensor_type == "gelsight":
                    # GelSight sensor data processing
                    if sensor_name not in gelsight_sensors:
                        gelsight_sensors[sensor_name] = {}
                    
                    gelsight_sensors[sensor_name][data_type] = current_data
                    
                    if data_type == "tactile_image":
                        tactile_image = current_data
                        
                        # Ensure correct image format
                        if isinstance(tactile_image, np.ndarray):
                            if tactile_image.ndim == 3 and tactile_image.shape[-1] == 3:
                                # BGR to RGB conversion
                                tactile_image = tactile_image[..., ::-1]
                                
                                # Log tactile image
                                rr.log(f"Tactile/GelSight/{sensor_name}/Image", 
                                      rr.Image(tactile_image))
                                
                                # Calculate and log image statistics
                                mean_brightness = np.mean(tactile_image)
                                contrast = np.std(tactile_image)
                                
                                rr.log(f"Tactile/GelSight/{sensor_name}/Stats/Brightness", 
                                      rr.Scalars(mean_brightness))
                                rr.log(f"Tactile/GelSight/{sensor_name}/Stats/Contrast", 
                                      rr.Scalars(contrast))
                    
                    elif data_type in ["frame_index", "recv_timestamp", "send_timestamp"]:
                        # Log metadata
                        metadata_value = current_data
                        if isinstance(metadata_value, (np.ndarray, torch.Tensor)):
                            metadata_value = float(metadata_value.item())
                        
                        rr.log(f"Tactile/GelSight/{sensor_name}/Metadata/{data_type}", 
                              rr.Scalars(metadata_value))
                
                # Log sensor metadata
                if data_type == "sensor_sn":
                    sensor_sn = str(current_data)
                    if sensor_type == "tac3d":
                        rr.log(f"Tactile/Tac3D/{sensor_name}/Info", 
                              rr.TextLog(f"Serial Number: {sensor_sn}"))
                    elif sensor_type == "gelsight":
                        rr.log(f"Tactile/GelSight/{sensor_name}/Info", 
                              rr.TextLog(f"Serial Number: {sensor_sn}"))
        
        # Comprehensive 3D visualization for each Tac3D sensor
        for sensor_name, sensor_data in tac3d_sensors.items():
            # 获取3D数据，支持多种可能的键名
            positions = sensor_data.get('3D_Positions')
            displacements = sensor_data.get('3D_Displacements')
            forces = sensor_data.get('3D_Forces')
            
            # 打印调试信息
            print(f"\n传感器 {sensor_name} 数据键值:")
            for key, value in sensor_data.items():
                shape_info = f"形状={value.shape}" if hasattr(value, 'shape') else "无形状"
                print(f"  - {key}: {type(value)}, {shape_info}")
            
            # Call enhanced Tac3D visualization function
            visualize_tac3d_points_and_vectors(
                sensor_name=f"Tactile/Tac3D/{sensor_name}",
                positions=positions,
                displacements=displacements,
                forces=forces,
                scale_displacement=5.0,
                scale_force=30.0
            )

    if save:
        if output_dir is not None:
            output_path = output_dir / f"{repo_id.replace('/', '_')}_episode_{episode_index}.rrd"
            rr.save(str(output_path))
            logging.info(f"Saved to {output_path}")
            return output_path

    if mode == "local":
        logging.info("Press Ctrl+C to terminate the visualizer.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun path/to/file.rrd` on your local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    repo_id = args.repo_id
    root = args.root
    tolerance_s = args.tolerance_s
    episode_index = args.episode_index

    logging.info("Loading dataset metadata only")
    # First load dataset without specifying episodes to check available episodes
    # Force use pyav backend to avoid torchcodec dependency
    dataset_meta = LeRobotDataset(
        repo_id, 
        root=root, 
        tolerance_s=tolerance_s,
        download_videos=False,  # Don't download videos for initial check
        video_backend="pyav"  # Force use pyav backend
    )
    
    # Check if the requested episode exists
    available_episodes = dataset_meta.num_episodes
    total_frames = dataset_meta.num_frames
    print(f"📊 数据集信息:")
    print(f"   - 总episode数量: {available_episodes} 个 (编号从 0 到 {available_episodes-1})")
    print(f"   - 总frame数量: {total_frames} 个")
    print(f"   - 平均每个episode: {total_frames//available_episodes} 个frames")
    
    if episode_index >= available_episodes:
        print(f"\n❌ 错误：请求的 episode {episode_index} 不存在！")
        print(f"   可用的episode编号范围: 0 到 {available_episodes-1}")
        print(f"   请使用 --episode-index 参数指定正确的episode编号")
        print(f"\n💡 示例命令:")
        print(f"   python lerobot/scripts/visualize_dataset_simple.py \\")
        print(f"       --repo-id {repo_id} \\")
        print(f"       --episode-index 0")
        return
    
    print(f"✅ 正在可视化 episode {episode_index}")
    
    # Now load the dataset with the specific episode
    dataset = LeRobotDataset(
        repo_id, 
        root=root, 
        tolerance_s=tolerance_s,
        episodes=[episode_index],  # Only load specific episode
        download_videos=True,  # Still need videos for visualization
        video_backend="pyav"  # Force use pyav backend
    )

    visualize_dataset_simple(
        dataset=dataset,
        episode_index=episode_index,
        mode=args.mode,
        web_port=args.web_port,
        ws_port=args.ws_port,
        save=bool(args.save),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main() 