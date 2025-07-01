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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Enhanced Tactile Sensor Support:
- Tac3D sensors: Now supports comprehensive 3D tactile visualization including:
  * Resultant forces and moments (x, y, z components)
  * 3D point cloud visualization (400 tactile sensing points)
  * 3D displacement and force vector fields
  * Statistical analysis of tactile data
  * Data formats: observation.tactile.tac3d.{name}.{field}
- GelSight sensors: RGB tactile images and statistical analysis
  * observation.tactile.gelsight.{name}.tactile_image
  * Brightness and contrast metrics
- Hierarchical data structure: observation.tactile.{sensor_type}.{name}.{field}

Examples:

- Visualize data stored on a local machine:
```
local$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data with advanced Tac3D tactile visualization:
```
local$ python lerobot/scripts/visualize_dataset.py \
    --repo-id your_dataset_with_tac3d \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = int(dataset.episode_data_index["from"][episode_index].item())
        to_idx = int(dataset.episode_data_index["to"][episode_index].item())
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def generate_tac3d_mesh_connectivity(nx: int = 20, ny: int = 20):
    """
    生成Tac3D传感器网格连接信息
    参考PyTac3D_Displayer.py的_GenConnect方法
    
    Args:
        nx: X方向网格数量
        ny: Y方向网格数量
        
    Returns:
        连接索引列表，每个三角形由3个顶点索引组成
    """
    connect = []
    for iy in range(ny-1):
        for ix in range(nx-1):
            idx = iy * nx + ix
            # 每个网格单元生成两个三角形
            connect.append([idx, idx+1, idx+nx])
            connect.append([idx+nx+1, idx+nx, idx+1])
    return np.array(connect)


def generate_tac3d_demo_data():
    """
    生成Tac3D演示数据（当实际数据全为零时使用）
    参考PyTac3D_Displayer.py的数据结构
    """
    # 生成20x20网格位置
    nx, ny = 20, 20
    x = np.linspace(-8, 8, nx)
    y = np.linspace(-8, 8, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # 生成径向力场（中心区域有较大的力）
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
    
    # 生成位移数据（与力成比例）
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
    """
    可视化Tac3D传感器的3D点云和向量场
    参考PyTac3D_Displayer.py的可视化方法，在Rerun中呈现
    
    Args:
        sensor_name: 传感器名称
        positions: 3D位置数据 (400, 3)
        displacements: 3D位移数据 (400, 3)，可选
        forces: 3D力数据 (400, 3)，可选
        scale_displacement: 位移向量的缩放因子（参考PyTac3D_Displayer: 5倍）
        scale_force: 力向量的缩放因子（参考PyTac3D_Displayer: 30倍）
    """
    if positions is None or positions.size == 0:
        return
        
    # 确保数据格式正确
    if positions.ndim != 2 or positions.shape[1] != 3:
        print(f"Warning: Invalid positions shape: {positions.shape}, expected (N, 3)")
        return
    
    # 检查数据是否全为零（传感器未正确初始化）
    positions_zero = np.all(positions == 0)
    forces_zero = forces is None or np.all(forces == 0) if forces is not None else True
    displacements_zero = displacements is None or np.all(displacements == 0) if displacements is not None else True
    
    if positions_zero and forces_zero and displacements_zero:
        # 如果所有数据都为零，使用演示数据并显示警告
        print(f"⚠️  Warning: Tac3D sensor {sensor_name} data is all zeros. Using demo data for visualization.")
        
        demo_positions, demo_displacements, demo_forces = generate_tac3d_demo_data()
        
        # 记录警告信息
        rr.log(f"tactile/tac3d/{sensor_name}/sensor_warning", 
               rr.TextLog("⚠️ DEMO DATA: Real sensor data is all zeros!\n"
                         "This indicates sensor initialization or calibration issues.\n"
                         "Showing demo data to demonstrate visualization capabilities."))
        
        # 使用演示数据进行可视化
        positions = demo_positions
        displacements = demo_displacements  
        forces = demo_forces
        
        # 降低演示数据的缩放因子以适应可视化
        scale_force = scale_force * 0.5
        scale_displacement = scale_displacement * 0.5
    
    try:
        # 1. 可视化3D触觉传感器表面网格（参考PyTac3D_Displayer的Mesh）
        if positions.shape[0] == 400:  # 标准20x20网格
            # 生成网格连接信息
            mesh_triangles = generate_tac3d_mesh_connectivity(20, 20)
            
            # 使用基于Z坐标的颜色映射
            z_min, z_max = float(positions[:, 2].min()), float(positions[:, 2].max())
            if z_max > z_min:
                z_normalized = (positions[:, 2] - z_min) / (z_max - z_min)
                # 蓝色到淡紫色的渐变（参考PyTac3D_Displayer的[150,150,230]）
                vertex_colors = np.zeros((len(positions), 3))
                vertex_colors[:, 0] = 0.4 + 0.4 * z_normalized  # 红色分量
                vertex_colors[:, 1] = 0.4 + 0.4 * z_normalized  # 绿色分量  
                vertex_colors[:, 2] = 0.7 + 0.3 * z_normalized  # 蓝色分量
            else:
                # 统一的淡紫色（类似PyTac3D_Displayer）
                vertex_colors = np.array([[0.6, 0.6, 0.9]] * len(positions))
            
            # 记录3D网格表面
            rr.log(f"tactile/tac3d/{sensor_name}/surface_mesh", 
                   rr.Mesh3D(
                       vertex_positions=positions,
                       triangle_indices=mesh_triangles,
                       vertex_colors=vertex_colors
                   ))
        
        # 2. 可视化3D触觉感知点（作为点云）
        rr.log(f"tactile/tac3d/{sensor_name}/sensing_points", 
               rr.Points3D(positions, colors=vertex_colors if 'vertex_colors' in locals() else None, radii=0.15))
        
        # 3. 可视化位移向量场（参考PyTac3D_Displayer的Displacements箭头）
        if displacements is not None and displacements.size > 0:
            if displacements.shape == positions.shape:
                # 计算位移幅度
                displacement_magnitudes = np.linalg.norm(displacements, axis=1)
                max_displacement = displacement_magnitudes.max()
                
                if max_displacement > 0:
                    # 过滤显著位移（避免噪声）
                    threshold = max_displacement * 0.05  # 5%阈值
                    significant_mask = displacement_magnitudes > threshold
                    
                    if np.any(significant_mask):
                        start_points = positions[significant_mask]
                        # 应用缩放因子（参考PyTac3D_Displayer: _scaleD = 5）
                        displacement_vectors = displacements[significant_mask] * scale_displacement
                        
                        # 绿色系颜色编码（位移大小）
                        disp_normalized = displacement_magnitudes[significant_mask] / max_displacement
                        vector_colors = np.zeros((len(start_points), 3))
                        vector_colors[:, 1] = 0.3 + 0.7 * disp_normalized  # 绿色主色调
                        vector_colors[:, 0] = 0.2 * (1 - disp_normalized)  # 少量红色对比
                        vector_colors[:, 2] = 0.1  # 少量蓝色
                        
                        # 记录位移箭头（参考PyTac3D_Displayer的arrsD）
                        rr.log(f"tactile/tac3d/{sensor_name}/displacement_arrows",
                               rr.Arrows3D(
                                   origins=start_points, 
                                   vectors=displacement_vectors, 
                                   colors=vector_colors,
                                   radii=0.05  # 箭头粗细
                               ))
                        
                        # 位移统计信息
                        rr.log(f"tactile/tac3d/{sensor_name}/displacement_stats", 
                               rr.TextLog(f"Max displacement: {max_displacement:.4f}mm, "
                                         f"Mean: {np.mean(displacement_magnitudes):.4f}mm, "
                                         f"Active points: {np.sum(significant_mask)}/400"))
        
        # 4. 可视化所有400个点的力向量场（参考PyTac3D_Displayer的Forces箭头）
        if forces is not None and forces.size > 0:
            if forces.shape == positions.shape and forces.shape[0] == 400:
                # 计算力幅度
                force_magnitudes = np.linalg.norm(forces, axis=1)
                max_force = force_magnitudes.max()
                mean_force = np.mean(force_magnitudes)
                
                # 显示所有400个点的力向量，不进行过滤
                # 应用缩放因子（参考PyTac3D_Displayer: _scaleF = 30）
                force_vectors = forces * scale_force
                
                # 基于力幅度的颜色编码（红色渐变）
                if max_force > 0:
                    force_normalized = force_magnitudes / max_force
                else:
                    force_normalized = np.zeros(400)
                
                # 创建20x20网格的颜色映射
                force_colors = np.zeros((400, 3))
                for i in range(400):
                    intensity = force_normalized[i]
                    # 红色主色调，强度越大越红
                    force_colors[i, 0] = 0.3 + 0.7 * intensity  # 红色分量 [0.3, 1.0]
                    force_colors[i, 1] = 0.1 * (1 - intensity)  # 绿色分量 [0.0, 0.1]
                    force_colors[i, 2] = 0.1 * (1 - intensity)  # 蓝色分量 [0.0, 0.1]
                
                # 记录所有400个点的力箭头（参考PyTac3D_Displayer的arrsF）
                rr.log(f"tactile/tac3d/{sensor_name}/force_arrows_all",
                       rr.Arrows3D(
                           origins=positions,  # 所有400个点作为起点
                           vectors=force_vectors,  # 所有400个力向量
                           colors=force_colors,  # 每个点的颜色
                           radii=0.05  # 箭头粗细
                       ))
                
                # 额外显示有显著力的点（用更粗的箭头突出显示）
                if max_force > 0:
                    threshold = max_force * 0.1  # 10%阈值
                    significant_mask = force_magnitudes > threshold
                    
                    if np.any(significant_mask):
                        significant_positions = positions[significant_mask]
                        significant_forces = force_vectors[significant_mask]
                        significant_colors = force_colors[significant_mask]
                        
                        # 显著力点用更粗更亮的箭头
                        bright_colors = np.copy(significant_colors)
                        bright_colors[:, 0] = np.minimum(bright_colors[:, 0] * 1.5, 1.0)  # 增强红色
                        
                        rr.log(f"tactile/tac3d/{sensor_name}/force_arrows_significant",
                               rr.Arrows3D(
                                   origins=significant_positions,
                                   vectors=significant_forces,
                                   colors=bright_colors,
                                   radii=0.12  # 更粗的箭头
                               ))
                
                # 力统计信息和网格信息
                active_points = np.sum(force_magnitudes > mean_force * 0.1)
                rr.log(f"tactile/tac3d/{sensor_name}/force_grid_stats", 
                       rr.TextLog(f"20x20 Force Grid Visualization:\n"
                                 f"Max force: {max_force:.4f}N\n"
                                 f"Mean force: {mean_force:.4f}N\n"
                                 f"Active points (>10% mean): {active_points}/400\n"
                                 f"Force scale factor: {scale_force}x"))
                
                # 可视化力分布的热力图（在XY平面投影）
                if positions.shape[0] == 400:  # 确保是20x20网格
                    # 重塑为20x20网格用于热力图显示
                    force_grid = force_magnitudes.reshape(20, 20)
                    
                    # 创建网格坐标
                    x_coords = positions[:, 0].reshape(20, 20)
                    y_coords = positions[:, 1].reshape(20, 20)
                    z_base = positions[:, 2].min()  # 使用最低Z坐标作为热力图基准面
                    
                    # 为热力图创建颜色映射
                    if max_force > 0:
                        normalized_grid = force_grid / max_force
                    else:
                        normalized_grid = np.zeros((20, 20))
                    
                    # 创建热力图的顶点和颜色
                    heatmap_points = []
                    heatmap_colors = []
                    for i in range(20):
                        for j in range(20):
                            heatmap_points.append([x_coords[i, j], y_coords[i, j], z_base - 1.0])
                            intensity = normalized_grid[i, j]
                            heatmap_colors.append([intensity, 0.0, 1.0 - intensity])  # 蓝到红渐变
                    
                    rr.log(f"tactile/tac3d/{sensor_name}/force_heatmap",
                           rr.Points3D(
                               positions=np.array(heatmap_points),
                               colors=np.array(heatmap_colors),
                               radii=0.8  # 较大的点形成热力图效果
                           ))
        
        # 5. 可视化传感器边界框（参考PyTac3D_Displayer的Box）
        if positions.size > 0:
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            center = (min_pos + max_pos) / 2
            size = max_pos - min_pos
            
            # 半透明边界框（参考PyTac3D_Displayer的Box alpha=0.03）
            box_corners = np.array([
                [min_pos[0], min_pos[1], min_pos[2]],  # 底面四个角
                [max_pos[0], min_pos[1], min_pos[2]],
                [max_pos[0], max_pos[1], min_pos[2]],
                [min_pos[0], max_pos[1], min_pos[2]],
                [min_pos[0], min_pos[1], max_pos[2]],  # 顶面四个角
                [max_pos[0], min_pos[1], max_pos[2]],
                [max_pos[0], max_pos[1], max_pos[2]],
                [min_pos[0], max_pos[1], max_pos[2]],
            ])
            
            # 边界框的线框
            box_lines = np.array([
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                [0, 4], [1, 5], [2, 6], [3, 7],  # 连接线
            ])
            
            rr.log(f"tactile/tac3d/{sensor_name}/sensor_bounds",
                   rr.LineStrips3D([box_corners[box_lines.flatten()]], colors=[0.3, 0.3, 0.3]))
            
            # 坐标轴（参考PyTac3D_Displayer的Axes）
            axis_length = np.max(size) * 0.3
            axes_origins = np.array([[center[0], center[1], min_pos[2]]] * 3)
            axes_vectors = np.array([
                [axis_length, 0, 0],  # X轴 - 红色
                [0, axis_length, 0],  # Y轴 - 绿色  
                [0, 0, axis_length],  # Z轴 - 蓝色
            ])
            axes_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            
            rr.log(f"tactile/tac3d/{sensor_name}/coordinate_axes",
                   rr.Arrows3D(origins=axes_origins, vectors=axes_vectors, colors=axes_colors))
            
            # 传感器尺寸信息
            rr.log(f"tactile/tac3d/{sensor_name}/sensor_info", 
                   rr.TextLog(f"Sensor dimensions: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} mm\n"
                             f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) mm\n"
                             f"Total sensing points: {len(positions)}"))
                             
    except Exception as e:
        print(f"Warning: Error visualizing Tac3D data for {sensor_name}: {e}")


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        # For distant viewing, just log and let user handle connection
        print(f"Starting server on web_port={web_port}, ws_port={ws_port}")
        print(f"Please use: rerun ws://localhost:{ws_port} from your local machine")

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["frame_index"][i].item())
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            # display each camera image
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalars(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

            if "next.done" in batch:
                rr.log("next.done", rr.Scalars(batch["next.done"][i].item()))

            if "next.reward" in batch:
                rr.log("next.reward", rr.Scalars(batch["next.reward"][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

            # Enhanced tactile sensor data visualization
            # 收集当前帧所有触觉传感器数据以便进行综合可视化
            tac3d_sensors = {}  # {sensor_name: {data_type: data}}
            gelsight_sensors = {}  # {sensor_name: {data_type: data}}
            
            for key in batch.keys():
                if key.startswith("observation.tactile."):
                    # 支持两种数据结构：
                    # 新结构：observation.tactile.{sensor_type}.{name}.{field}
                    # 旧结构：observation.tactile.{name}.{field}
                    parts = key.split(".")
                    
                    if len(parts) >= 5:
                        # 新的分层结构：observation.tactile.{sensor_type}.{name}.{field}
                        sensor_type = parts[2]  # "gelsight" 或 "tac3d"
                        sensor_name = parts[3]  # "main_gripper0", "main_gripper1", etc.
                        data_type = parts[4]    # "tactile_image", "resultant_force", etc.
                    elif len(parts) >= 4:
                        # 旧的扁平结构：observation.tactile.{name}.{field}
                        sensor_name = parts[2]  # "left_gripper", "right_gripper", etc.
                        data_type = parts[3]    # "tactile_image", "resultant_force", etc.
                        
                        # 根据数据类型推断传感器类型
                        if data_type in ["tactile_image"]:
                            sensor_type = "gelsight"
                        elif data_type in ["resultant_force", "resultant_moment", "positions_3d", "forces_3d", "displacements_3d", "sensor_sn", "frame_index", "send_timestamp", "recv_timestamp"]:
                            sensor_type = "tac3d"
                        else:
                            sensor_type = "unknown"
                    else:
                        continue  # 跳过不符合格式的键

                    # 获取当前数据
                    current_data = batch[key][i]
                    if isinstance(current_data, torch.Tensor):
                        current_data = current_data.numpy()

                    # 根据传感器类型分别处理
                    if sensor_type == "tac3d":
                        # Tac3D传感器数据处理
                        if sensor_name not in tac3d_sensors:
                            tac3d_sensors[sensor_name] = {}
                        
                        # 字段名映射
                        field_mapping = {
                            'positions_3d': '3D_Positions',
                            'displacements_3d': '3D_Displacements',
                            'forces_3d': '3D_Forces',
                            'resultant_force': '3D_ResultantForce',
                            'resultant_moment': '3D_ResultantMoment',
                            'sensor_sn': 'SN',
                            'frame_index': 'index',
                            'send_timestamp': 'sendTimestamp',
                            'recv_timestamp': 'recvTimestamp'
                        }
                        
                        # 转换字段名
                        data_type_mapped = field_mapping.get(data_type, data_type)
                        tac3d_sensors[sensor_name][data_type_mapped] = current_data
                        
                        # 处理力和力矩数据
                        if data_type == 'resultant_force' or data_type_mapped == '3D_ResultantForce':
                            force = current_data
                            if force.ndim == 2 and force.shape[0] == 1:
                                force = force[0]
                            
                            if force.size >= 3:
                                # 记录xyz方向的力分量
                                for dim_idx, force_component in enumerate(force[:3]):
                                    axis_name = ['x', 'y', 'z'][dim_idx]
                                    rr.log(f"Tactile/Tac3D/{sensor_name}/Forces/Component/{axis_name}", 
                                          rr.Scalars(force_component))
                                
                                # 记录合力幅度
                                force_magnitude = np.linalg.norm(force[:3])
                                rr.log(f"Tactile/Tac3D/{sensor_name}/Forces/Magnitude", 
                                      rr.Scalars(force_magnitude))
                                
                        elif data_type == 'resultant_moment' or data_type_mapped == '3D_ResultantMoment':
                            moment = current_data
                            if moment.ndim == 2 and moment.shape[0] == 1:
                                moment = moment[0]
                            
                            if moment.size >= 3:
                                # 记录xyz方向的力矩分量
                                for dim_idx, moment_component in enumerate(moment[:3]):
                                    axis_name = ['x', 'y', 'z'][dim_idx]
                                    rr.log(f"Tactile/Tac3D/{sensor_name}/Moments/Component/{axis_name}", 
                                          rr.Scalars(moment_component))
                                
                                # 记录合力矩幅度
                                moment_magnitude = np.linalg.norm(moment[:3])
                                rr.log(f"Tactile/Tac3D/{sensor_name}/Moments/Magnitude", 
                                      rr.Scalars(moment_magnitude))

                    elif sensor_type == "gelsight":
                        # GelSight传感器数据处理
                        if sensor_name not in gelsight_sensors:
                            gelsight_sensors[sensor_name] = {}
                        
                        gelsight_sensors[sensor_name][data_type] = current_data
                        
                        if data_type == "tactile_image":
                            tactile_image = current_data
                            
                            # 确保图像格式正确
                            if isinstance(tactile_image, np.ndarray):
                                if tactile_image.ndim == 3 and tactile_image.shape[-1] == 3:
                                    # BGR转RGB
                                    tactile_image = tactile_image[..., ::-1]
                                    
                                    # 记录触觉图像
                                    rr.log(f"Tactile/GelSight/{sensor_name}/Image", 
                                          rr.Image(tactile_image))
                                    
                                    # 计算并记录图像统计信息
                                    mean_brightness = np.mean(tactile_image)
                                    contrast = np.std(tactile_image)
                                    
                                    rr.log(f"Tactile/GelSight/{sensor_name}/Stats/Brightness", 
                                          rr.Scalars(mean_brightness))
                                    rr.log(f"Tactile/GelSight/{sensor_name}/Stats/Contrast", 
                                          rr.Scalars(contrast))
                        
                        elif data_type in ["frame_index", "recv_timestamp", "send_timestamp"]:
                            # 记录元数据
                            metadata_value = current_data
                            if isinstance(metadata_value, (np.ndarray, torch.Tensor)):
                                metadata_value = float(metadata_value.item())
                            
                            rr.log(f"Tactile/GelSight/{sensor_name}/Metadata/{data_type}", 
                                  rr.Scalars(metadata_value))
                    
                    # 记录传感器元数据
                    if data_type == "sensor_sn":
                        sensor_sn = str(current_data)
                        if sensor_type == "tac3d":
                            rr.log(f"Tactile/Tac3D/{sensor_name}/Info", 
                                  rr.TextLog(f"Serial Number: {sensor_sn}"))
                        elif sensor_type == "gelsight":
                            rr.log(f"Tactile/GelSight/{sensor_name}/Info", 
                                  rr.TextLog(f"Serial Number: {sensor_sn}"))
            
            # 对每个Tac3D传感器进行综合3D可视化
            for sensor_name, sensor_data in tac3d_sensors.items():
                positions = sensor_data.get('3D_Positions')
                displacements = sensor_data.get('3D_Displacements')
                forces = sensor_data.get('3D_Forces')
                
                # 调用增强的Tac3D可视化函数
                visualize_tac3d_points_and_vectors(
                    sensor_name=f"Tactile/Tac3D/{sensor_name}",  # 更新可视化路径
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of processes of Dataloader for loading the data. Set to 0 to avoid multiprocessing issues.",
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
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    logging.info("Loading dataset")
    # Force use pyav backend to avoid torchcodec dependency
    dataset = LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s, video_backend="pyav")
    
    # Check if the requested episode exists
    episode_index = args.episode_index
    available_episodes = dataset.num_episodes
    total_frames = dataset.num_frames
    print(f"📊 数据集信息:")
    print(f"   - 总episode数量: {available_episodes} 个 (编号从 0 到 {available_episodes-1})")
    print(f"   - 总frame数量: {total_frames} 个")
    print(f"   - 平均每个episode: {total_frames//available_episodes} 个frames")
    
    if episode_index >= available_episodes:
        print(f"\n❌ 错误：请求的 episode {episode_index} 不存在！")
        print(f"   可用的episode编号范围: 0 到 {available_episodes-1}")
        print(f"   请使用 --episode-index 参数指定正确的episode编号")
        print(f"\n💡 示例命令:")
        print(f"   python lerobot/scripts/visualize_dataset.py \\")
        print(f"       --repo-id {repo_id} \\")
        print(f"       --episode-index 0")
        return
    
    print(f"✅ 正在可视化 episode {episode_index}")

    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
