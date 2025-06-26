#!/usr/bin/env python3
"""
检查数据集中保存的Tac3D数据
分析数据结构、格式和内容
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def analyze_tac3d_keys(batch: Dict[str, Any]) -> Dict[str, list]:
    """分析批次中的Tac3D相关键"""
    tac3d_keys = {
        'new_format': [],      # observation.tactile.tac3d.{name}.{field}
        'old_format': [],      # observation.tactile.{name}.{field}
        'other_tactile': []    # 其他触觉相关键
    }
    
    for key in batch.keys():
        if key.startswith("observation.tactile."):
            parts = key.split(".")
            
            if len(parts) >= 5 and parts[2] == "tac3d":
                # 新格式：observation.tactile.tac3d.{name}.{field}
                tac3d_keys['new_format'].append(key)
            elif len(parts) >= 4 and parts[2] != "gelsight":
                # 旧格式：observation.tactile.{name}.{field}
                tac3d_keys['old_format'].append(key)
            else:
                # 其他触觉数据
                tac3d_keys['other_tactile'].append(key)
    
    return tac3d_keys

def print_tensor_info(name: str, data: torch.Tensor, level: int = 0):
    """打印张量的详细信息"""
    indent = "  " * level
    print(f"{indent}{name}:")
    print(f"{indent}  - 类型: {type(data)}")
    print(f"{indent}  - 形状: {data.shape}")
    print(f"{indent}  - 数据类型: {data.dtype}")
    
    if data.numel() > 0:
        if data.dtype in [torch.float32, torch.float64]:
            print(f"{indent}  - 数值范围: [{data.min().item():.6f}, {data.max().item():.6f}]")
            print(f"{indent}  - 平均值: {data.mean().item():.6f}")
            print(f"{indent}  - 标准差: {data.std().item():.6f}")
            
            # 检查是否全为零
            if torch.all(data == 0):
                print(f"{indent}  - ⚠️  所有值都为零！")
            elif torch.sum(data != 0).item() < data.numel() * 0.1:
                non_zero_count = torch.sum(data != 0).item()
                print(f"{indent}  - ⚠️  大部分值为零 (非零: {non_zero_count}/{data.numel()})")
        
        # 显示部分数据样本
        if data.numel() <= 10:
            print(f"{indent}  - 数据: {data.flatten()}")
        else:
            print(f"{indent}  - 样本: {data.flatten()[:5]}...")
    else:
        print(f"{indent}  - ⚠️  张量为空")

def analyze_tac3d_data_structure(dataset_path: str):
    """分析Tac3D数据集的数据结构"""
    print("🔍 开始检查Tac3D数据集...")
    print(f"数据集路径: {dataset_path}")
    print("=" * 80)
    
    try:
        # 加载数据集
        dataset = LeRobotDataset(dataset_path)
        print(f"✅ 数据集加载成功")
        print(f"数据集大小: {len(dataset)} 个样本")
        print(f"Episode数量: {dataset.num_episodes}")
        print()
        
        # 检查数据集元数据
        print("📊 数据集元数据:")
        print(f"  FPS: {dataset.fps}")
        print(f"  总帧数: {dataset.num_frames}")
        print(f"  相机键: {dataset.meta.camera_keys}")
        print(f"  所有键: {len(dataset.meta.features)} 个")
        print()
        
        # 获取第一个样本进行分析
        print("🎯 分析第一个样本...")
        sample = dataset[0]
        
        # 分析Tac3D相关键
        tac3d_keys = analyze_tac3d_keys(sample)
        
        print("📋 触觉数据键分类:")
        print(f"  新格式键 (observation.tactile.tac3d.*): {len(tac3d_keys['new_format'])}")
        for key in sorted(tac3d_keys['new_format']):
            print(f"    - {key}")
        
        print(f"  旧格式键 (observation.tactile.*): {len(tac3d_keys['old_format'])}")
        for key in sorted(tac3d_keys['old_format']):
            print(f"    - {key}")
            
        print(f"  其他触觉键: {len(tac3d_keys['other_tactile'])}")
        for key in sorted(tac3d_keys['other_tactile']):
            print(f"    - {key}")
        print()
        
        # 详细分析Tac3D数据
        all_tac3d_keys = tac3d_keys['new_format'] + tac3d_keys['old_format']
        
        if all_tac3d_keys:
            print("📈 Tac3D数据详细分析:")
            print("-" * 60)
            
            # 按传感器名称分组
            sensor_data = {}
            for key in all_tac3d_keys:
                parts = key.split(".")
                if len(parts) >= 5 and parts[2] == "tac3d":
                    # 新格式
                    sensor_name = parts[3]
                    field_name = parts[4]
                elif len(parts) >= 4:
                    # 旧格式
                    sensor_name = parts[2]
                    field_name = parts[3]
                else:
                    continue
                    
                if sensor_name not in sensor_data:
                    sensor_data[sensor_name] = {}
                sensor_data[sensor_name][field_name] = key
            
            # 分析每个传感器的数据
            for sensor_name, fields in sensor_data.items():
                print(f"\n🤖 传感器: {sensor_name}")
                print(f"  字段数量: {len(fields)}")
                
                for field_name, key in sorted(fields.items()):
                    data = sample[key]
                    print(f"\n  📊 {field_name} ({key}):")
                    print_tensor_info(field_name, data, level=2)
        
        else:
            print("❌ 未找到Tac3D数据！")
            print("可能的原因:")
            print("  1. 数据集中没有触觉传感器数据")
            print("  2. 触觉数据使用了不同的键名格式")
            print("  3. 数据集损坏或格式不正确")
            
            # 显示所有可用的键以便调试
            print(f"\n🔧 所有可用键 (前50个):")
            all_keys = sorted(sample.keys())
            for i, key in enumerate(all_keys[:50]):
                print(f"  {i+1:2d}. {key}")
            if len(all_keys) > 50:
                print(f"  ... 还有 {len(all_keys) - 50} 个键")
        
        # 检查多个样本以验证数据一致性
        if len(dataset) > 1:
            print(f"\n🔄 数据一致性检查 (检查前5个样本)...")
            num_samples_to_check = min(5, len(dataset))
            
            for i in range(1, num_samples_to_check):
                sample_i = dataset[i]
                tac3d_keys_i = analyze_tac3d_keys(sample_i)
                
                if tac3d_keys_i['new_format'] != tac3d_keys['new_format'] or \
                   tac3d_keys_i['old_format'] != tac3d_keys['old_format']:
                    print(f"  ⚠️  样本 {i} 的键结构与样本 0 不同")
                else:
                    print(f"  ✅ 样本 {i} 键结构一致")
                    
                    # 检查数据值
                    for key in all_tac3d_keys[:3]:  # 只检查前几个键
                        if key in sample_i:
                            data_0 = sample[key]
                            data_i = sample_i[key]
                            
                            if data_0.shape != data_i.shape:
                                print(f"    ⚠️  {key} 形状不一致: {data_0.shape} vs {data_i.shape}")
                            elif torch.allclose(data_0, data_i, atol=1e-6):
                                print(f"    ⚠️  {key} 数值完全相同 (可能未更新)")
        
        print(f"\n✅ 数据集检查完成")
        
    except Exception as e:
        print(f"❌ 检查数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 故障排除建议:")
        print(f"  1. 检查数据集路径是否正确: {dataset_path}")
        print(f"  2. 确认数据集是LeRobot格式")
        print(f"  3. 检查数据集文件是否完整")

def check_episode_data(dataset_path: str, episode_index: int = 0):
    """检查特定episode的数据"""
    print(f"\n🎬 检查Episode {episode_index}的数据...")
    
    try:
        dataset = LeRobotDataset(dataset_path)
        
        if episode_index >= dataset.num_episodes:
            print(f"❌ Episode {episode_index} 不存在 (总共 {dataset.num_episodes} 个episodes)")
            return
        
        # 获取episode的帧范围
        from_idx = int(dataset.episode_data_index["from"][episode_index].item())
        to_idx = int(dataset.episode_data_index["to"][episode_index].item())
        num_frames = to_idx - from_idx
        
        print(f"Episode {episode_index}:")
        print(f"  帧范围: {from_idx} - {to_idx-1}")
        print(f"  帧数量: {num_frames}")
        
        # 检查前几帧的Tac3D数据变化
        print(f"\n📊 前5帧的Tac3D数据变化:")
        
        for frame_idx in range(min(5, num_frames)):
            global_idx = from_idx + frame_idx
            sample = dataset[global_idx]
            
            print(f"\n  帧 {frame_idx} (全局索引 {global_idx}):")
            
            # 查找Tac3D force数据
            force_keys = [k for k in sample.keys() if 'resultant_force' in k and 'tactile' in k]
            
            for key in force_keys:
                force_data = sample[key]
                if isinstance(force_data, torch.Tensor) and force_data.numel() > 0:
                    magnitude = torch.norm(force_data).item()
                    print(f"    {key}: {force_data.numpy()} (大小: {magnitude:.6f})")
                else:
                    print(f"    {key}: 无数据")
    
    except Exception as e:
        print(f"❌ 检查episode数据时出错: {e}")

if __name__ == "__main__":
    import os
    
    # 使用完整路径
    dataset_path = "/home/user/.cache/huggingface/lerobot/user/test_two2"
    
    print(f"检查数据集路径: {dataset_path}")
    
    if not Path(dataset_path).exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        print("请确认数据集路径是否正确")
        exit(1)
    
    # 分析数据集结构
    analyze_tac3d_data_structure(dataset_path)
    
    # 检查episode数据
    check_episode_data(dataset_path, episode_index=0) 