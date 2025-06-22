#!/usr/bin/env python3
"""
测试GelSight数据可视化功能

本脚本创建一个模拟的LeRobot数据集，包含GelSight触觉数据，
然后测试visualize_dataset.py脚本的可视化功能。
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import h5py
from PIL import Image

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

def create_mock_gelsight_dataset():
    """创建包含GelSight数据的模拟数据集"""
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    dataset_dir = temp_dir / "test_gelsight_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集参数
    num_episodes = 2
    episode_length = 10
    image_height, image_width = 240, 320
    
    print(f"创建模拟数据集在: {dataset_dir}")
    
    # 创建数据集元数据
    meta_path = dataset_dir / "meta.json"
    meta_data = {
        "robot_type": "test_robot",
        "fps": 30,
        "video": False,
        "features": {
            "observation.tactile.left_gripper.tactile_image": {
                "dtype": "uint8",
                "shape": [image_height, image_width, 3],
                "names": ["height", "width", "channel"]
            },
            "observation.tactile.right_gripper.tactile_image": {
                "dtype": "uint8", 
                "shape": [image_height, image_width, 3],
                "names": ["height", "width", "channel"]
            },
            "observation.tactile.left_gripper.sensor_sn": {
                "dtype": "string",
                "shape": [],
                "names": []
            },
            "observation.tactile.right_gripper.sensor_sn": {
                "dtype": "string",
                "shape": [],
                "names": []
            },
            "observation.tactile.left_gripper.frame_index": {
                "dtype": "int64",
                "shape": [],
                "names": []
            },
            "observation.tactile.right_gripper.frame_index": {
                "dtype": "int64",
                "shape": [],
                "names": []
            },
            "observation.tactile.left_gripper.send_timestamp": {
                "dtype": "float64",
                "shape": [],
                "names": []
            },
            "observation.tactile.right_gripper.send_timestamp": {
                "dtype": "float64",
                "shape": [],
                "names": []
            },
            "observation.tactile.left_gripper.recv_timestamp": {
                "dtype": "float64",
                "shape": [],
                "names": []
            },
            "observation.tactile.right_gripper.recv_timestamp": {
                "dtype": "float64",
                "shape": [],
                "names": []
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [],
                "names": []
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [],
                "names": []
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [],
                "names": []
            },
            "next.done": {
                "dtype": "bool",
                "shape": [],
                "names": []
            },
            "next.reward": {
                "dtype": "float32",
                "shape": [],
                "names": []
            },
            "next.success": {
                "dtype": "bool",
                "shape": [],
                "names": []
            }
        },
        "total_episodes": num_episodes,
        "total_frames": num_episodes * episode_length,
        "camera_keys": [],  # 没有相机
        "tactile_keys": [
            "observation.tactile.left_gripper.tactile_image",
            "observation.tactile.right_gripper.tactile_image"
        ]
    }
    
    import json
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    # 创建必需的info.json文件
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    
    info_path = meta_dir / "info.json"
    info_data = {
        "robot_type": "test_robot",
        "robot_name": "Test Robot for GelSight",
        "fps": 30,
        "video": False,
        "repo_tags": ["tactile", "gelsight", "test"],
        "license": "apache-2.0",
        "description": "Test dataset for GelSight tactile sensor visualization"
    }
    
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=2)
    
    # 创建数据文件
    data_path = dataset_dir / "data.hdf5"
    
    total_frames = num_episodes * episode_length
    
    with h5py.File(data_path, 'w') as f:
        # 创建数据集
        
        # 触觉图像数据
        left_tactile_ds = f.create_dataset(
            "observation.tactile.left_gripper.tactile_image",
            (total_frames, image_height, image_width, 3),
            dtype=np.uint8
        )
        right_tactile_ds = f.create_dataset(
            "observation.tactile.right_gripper.tactile_image", 
            (total_frames, image_height, image_width, 3),
            dtype=np.uint8
        )
        
        # 触觉元数据
        left_sn_ds = f.create_dataset(
            "observation.tactile.left_gripper.sensor_sn",
            (total_frames,),
            dtype=h5py.string_dtype()
        )
        right_sn_ds = f.create_dataset(
            "observation.tactile.right_gripper.sensor_sn",
            (total_frames,),
            dtype=h5py.string_dtype()
        )
        
        left_frame_ds = f.create_dataset(
            "observation.tactile.left_gripper.frame_index",
            (total_frames,),
            dtype=np.int64
        )
        right_frame_ds = f.create_dataset(
            "observation.tactile.right_gripper.frame_index",
            (total_frames,),
            dtype=np.int64
        )
        
        left_send_ts_ds = f.create_dataset(
            "observation.tactile.left_gripper.send_timestamp",
            (total_frames,),
            dtype=np.float64
        )
        right_send_ts_ds = f.create_dataset(
            "observation.tactile.right_gripper.send_timestamp",
            (total_frames,),
            dtype=np.float64
        )
        
        left_recv_ts_ds = f.create_dataset(
            "observation.tactile.left_gripper.recv_timestamp",
            (total_frames,),
            dtype=np.float64
        )
        right_recv_ts_ds = f.create_dataset(
            "observation.tactile.right_gripper.recv_timestamp",
            (total_frames,),
            dtype=np.float64
        )
        
        # 其他必需数据
        action_ds = f.create_dataset("action", (total_frames, 6), dtype=np.float32)
        state_ds = f.create_dataset("observation.state", (total_frames, 6), dtype=np.float32)
        episode_index_ds = f.create_dataset("episode_index", (total_frames,), dtype=np.int64)
        frame_index_ds = f.create_dataset("frame_index", (total_frames,), dtype=np.int64)
        timestamp_ds = f.create_dataset("timestamp", (total_frames,), dtype=np.float64)
        done_ds = f.create_dataset("next.done", (total_frames,), dtype=bool)
        reward_ds = f.create_dataset("next.reward", (total_frames,), dtype=np.float32)
        success_ds = f.create_dataset("next.success", (total_frames,), dtype=bool)
        
        # 填充数据
        frame_idx = 0
        for episode_idx in range(num_episodes):
            print(f"  生成episode {episode_idx}")
            
            for step in range(episode_length):
                current_time = time.time() + frame_idx * 0.033  # 30fps
                
                # 生成模拟的触觉图像
                # 左手图像 - 渐变背景 + 随机噪声
                left_image = generate_tactile_image(
                    image_height, image_width, 
                    pattern="gradient", 
                    frame=frame_idx,
                    sensor="left"
                )
                
                # 右手图像 - 圆形模式 + 压力模拟
                right_image = generate_tactile_image(
                    image_height, image_width,
                    pattern="circle",
                    frame=frame_idx, 
                    sensor="right"
                )
                
                # 存储触觉数据
                left_tactile_ds[frame_idx] = left_image
                right_tactile_ds[frame_idx] = right_image
                
                # 触觉元数据
                left_sn_ds[frame_idx] = "GelSight_Left_001"
                right_sn_ds[frame_idx] = "GelSight_Right_002"
                left_frame_ds[frame_idx] = frame_idx
                right_frame_ds[frame_idx] = frame_idx
                left_send_ts_ds[frame_idx] = current_time
                right_send_ts_ds[frame_idx] = current_time
                left_recv_ts_ds[frame_idx] = current_time + 0.001
                right_recv_ts_ds[frame_idx] = current_time + 0.001
                
                # 其他数据
                action_ds[frame_idx] = np.random.randn(6) * 0.1
                state_ds[frame_idx] = np.random.randn(6) * 0.1
                episode_index_ds[frame_idx] = episode_idx
                frame_index_ds[frame_idx] = frame_idx
                timestamp_ds[frame_idx] = current_time
                done_ds[frame_idx] = (step == episode_length - 1)
                reward_ds[frame_idx] = np.random.random()
                success_ds[frame_idx] = (step == episode_length - 1)
                
                frame_idx += 1
    
    print(f"✓ 模拟数据集创建完成: {dataset_dir}")
    return dataset_dir


def generate_tactile_image(height, width, pattern="gradient", frame=0, sensor="left"):
    """生成模拟的触觉图像"""
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if pattern == "gradient":
        # 渐变背景
        for i in range(height):
            for j in range(width):
                # 基于位置的渐变
                r = int(255 * i / height)
                g = int(255 * j / width)
                b = int(255 * (i + j) / (height + width))
                
                # 添加时间变化
                time_factor = np.sin(frame * 0.1) * 0.3 + 0.7
                
                image[i, j] = [
                    int(r * time_factor),
                    int(g * time_factor), 
                    int(b * time_factor)
                ]
    
    elif pattern == "circle":
        # 圆形压力模式
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        
        for i in range(height):
            for j in range(width):
                # 距离中心的距离
                dist = np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
                
                if dist < radius:
                    # 圆形内部，模拟压力分布
                    pressure = 1.0 - (dist / radius)
                    
                    # 时间变化的压力
                    time_pressure = np.sin(frame * 0.2) * 0.5 + 0.5
                    final_pressure = pressure * time_pressure
                    
                    image[i, j] = [
                        int(255 * final_pressure),
                        int(128 * final_pressure),
                        int(64 * final_pressure)
                    ]
                else:
                    # 背景
                    image[i, j] = [50, 50, 50]
    
    # 添加随机噪声
    noise = np.random.randint(-20, 21, (height, width, 3))
    image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    
    return image


def test_visualization(dataset_path):
    """测试可视化功能"""
    print(f"\n测试可视化功能...")
    
    try:
        # 尝试加载数据集
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        # 使用root参数指定本地路径
        dataset = LeRobotDataset(repo_id="test_gelsight_dataset", root=str(dataset_path.parent))
        print(f"✓ 数据集加载成功")
        print(f"  总episodes: {dataset.total_episodes}")
        print(f"  总frames: {dataset.total_frames}")
        
        # 检查触觉keys
        if hasattr(dataset.meta, 'tactile_keys'):
            print(f"  触觉keys: {dataset.meta.tactile_keys}")
        else:
            print(f"  触觉keys: 未定义")
        
        # 测试单个数据点
        sample = dataset[0]
        print(f"\n✓ 数据采样测试:")
        for key, value in sample.items():
            if "tactile" in key:
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} {value.dtype}")
                else:
                    print(f"  {key}: {type(value)} {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🖼️  GelSight数据可视化测试")
    print("=" * 50)
    
    try:
        # 1. 创建模拟数据集
        dataset_path = create_mock_gelsight_dataset()
        
        # 2. 测试数据集加载
        if not test_visualization(dataset_path):
            return 1
        
        # 3. 提供可视化命令
        print(f"\n🎯 测试可视化命令:")
        print(f"python lerobot/scripts/visualize_dataset.py \\")
        print(f"    --repo-id {dataset_path} \\")
        print(f"    --episode-index 0 \\")
        print(f"    --mode local")
        
        print(f"\n💡 期望看到的可视化内容:")
        print(f"  📊 触觉数据:")
        print(f"    - tactile/left_gripper/tactile_image: 渐变触觉图像")
        print(f"    - tactile/right_gripper/tactile_image: 圆形压力图像") 
        print(f"    - tactile/*/image_stats/*: RGB通道统计")
        print(f"    - tactile/*/metadata/*: 传感器元数据")
        
        print(f"\n📁 数据集位置: {dataset_path}")
        print(f"⚠️  注意: 这是临时数据集，程序结束后会被清理")
        
        # 4. 可选：直接运行可视化
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--run-viz", action="store_true", help="直接运行可视化")
        args, _ = parser.parse_known_args()
        
        if args.run_viz:
            print(f"\n🚀 启动可视化...")
            import subprocess
            
            cmd = [
                "python", "lerobot/scripts/visualize_dataset.py",
                "--repo-id", str(dataset_path),
                "--episode-index", "0",
                "--mode", "local"
            ]
            
            subprocess.run(cmd)
        else:
            print(f"\n💡 添加 --run-viz 参数可直接运行可视化")
        
        return 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code) 