#!/usr/bin/env python3
"""
全面检查数据集中所有Tac3D数据
分析整个数据集的Tac3D数据分布和变化
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def comprehensive_tac3d_analysis(dataset_path: str):
    """全面分析所有Tac3D数据"""
    print("🔍 开始全面检查Tac3D数据集...")
    print(f"数据集路径: {dataset_path}")
    print("=" * 80)
    
    try:
        # 加载数据集
        dataset = LeRobotDataset(dataset_path)
        print(f"✅ 数据集加载成功")
        print(f"数据集大小: {len(dataset)} 个样本")
        print(f"Episode数量: {dataset.num_episodes}")
        print()
        
        # 找到所有Tac3D数据键
        sample = dataset[0]
        tac3d_keys = []
        force_keys = []
        moment_keys = []
        
        for key in sample.keys():
            if "tactile" in key and "tac3d" in key:
                tac3d_keys.append(key)
                if "resultant_force" in key:
                    force_keys.append(key)
                elif "resultant_moment" in key:
                    moment_keys.append(key)
        
        print(f"📊 找到 {len(tac3d_keys)} 个Tac3D数据键")
        print(f"   其中合力键: {len(force_keys)} 个")
        print(f"   合力矩键: {len(moment_keys)} 个")
        print()
        
        if not tac3d_keys:
            print("❌ 未找到Tac3D数据键")
            return
        
        # 分析所有样本的数据
        print("🔄 分析所有样本的Tac3D数据...")
        
        # 统计数据
        stats = {
            'non_zero_samples': 0,           # 非零样本数量
            'total_samples': len(dataset),   # 总样本数
            'force_stats': {},               # 合力统计
            'moment_stats': {},              # 合力矩统计
            'timestamp_stats': {},           # 时间戳统计
            'force_history': [],             # 合力历史数据
            'moment_history': [],            # 合力矩历史数据
            'non_zero_indices': [],          # 非零数据的索引
        }
        
        # 初始化统计数据结构
        for key in force_keys:
            stats['force_stats'][key] = {
                'non_zero_count': 0,
                'max_magnitude': 0.0,
                'values': []
            }
        
        for key in moment_keys:
            stats['moment_stats'][key] = {
                'non_zero_count': 0,
                'max_magnitude': 0.0,
                'values': []
            }
        
        # 遍历所有样本
        print("正在分析所有样本...")
        for i in range(len(dataset)):
            if i % 50 == 0:
                print(f"  进度: {i}/{len(dataset)} ({i/len(dataset)*100:.1f}%)")
            
            sample = dataset[i]
            sample_has_non_zero = False
            
            # 检查合力数据
            for key in force_keys:
                force_data = sample[key]
                magnitude = torch.norm(force_data).item()
                stats['force_stats'][key]['values'].append(magnitude)
                
                if magnitude > 1e-6:  # 非零阈值
                    stats['force_stats'][key]['non_zero_count'] += 1
                    stats['force_stats'][key]['max_magnitude'] = max(
                        stats['force_stats'][key]['max_magnitude'], magnitude
                    )
                    sample_has_non_zero = True
                    
                    # 记录具体数值
                    if magnitude > 0.001:  # 记录显著非零值
                        stats['force_history'].append({
                            'index': i,
                            'key': key,
                            'force': force_data.numpy(),
                            'magnitude': magnitude
                        })
            
            # 检查合力矩数据
            for key in moment_keys:
                moment_data = sample[key]
                magnitude = torch.norm(moment_data).item()
                stats['moment_stats'][key]['values'].append(magnitude)
                
                if magnitude > 1e-6:  # 非零阈值
                    stats['moment_stats'][key]['non_zero_count'] += 1
                    stats['moment_stats'][key]['max_magnitude'] = max(
                        stats['moment_stats'][key]['max_magnitude'], magnitude
                    )
                    sample_has_non_zero = True
                    
                    # 记录具体数值
                    if magnitude > 0.001:  # 记录显著非零值
                        stats['moment_history'].append({
                            'index': i,
                            'key': key,
                            'moment': moment_data.numpy(),
                            'magnitude': magnitude
                        })
            
            if sample_has_non_zero:
                stats['non_zero_samples'] += 1
                stats['non_zero_indices'].append(i)
        
        print(f"  分析完成: {len(dataset)}/{len(dataset)} (100.0%)")
        print()
        
        # 打印统计结果
        print("📈 数据统计结果:")
        print("-" * 60)
        print(f"总样本数: {stats['total_samples']}")
        print(f"非零样本数: {stats['non_zero_samples']}")
        print(f"非零比例: {stats['non_zero_samples']/stats['total_samples']*100:.2f}%")
        print()
        
        # 合力统计
        print("🔸 合力数据统计:")
        for key, stat in stats['force_stats'].items():
            sensor_name = key.split('.')[-2] if '.' in key else 'unknown'
            print(f"  {sensor_name}:")
            print(f"    非零样本: {stat['non_zero_count']}/{stats['total_samples']} ({stat['non_zero_count']/stats['total_samples']*100:.2f}%)")
            print(f"    最大力大小: {stat['max_magnitude']:.6f} N")
            
            # 计算统计信息
            values = np.array(stat['values'])
            if len(values) > 0:
                print(f"    平均力大小: {np.mean(values):.6f} N")
                print(f"    标准差: {np.std(values):.6f} N")
                print(f"    数值分布: min={np.min(values):.6f}, max={np.max(values):.6f}")
        print()
        
        # 合力矩统计
        print("🔸 合力矩数据统计:")
        for key, stat in stats['moment_stats'].items():
            sensor_name = key.split('.')[-2] if '.' in key else 'unknown'
            print(f"  {sensor_name}:")
            print(f"    非零样本: {stat['non_zero_count']}/{stats['total_samples']} ({stat['non_zero_count']/stats['total_samples']*100:.2f}%)")
            print(f"    最大力矩大小: {stat['max_magnitude']:.6f} N·m")
            
            # 计算统计信息
            values = np.array(stat['values'])
            if len(values) > 0:
                print(f"    平均力矩大小: {np.mean(values):.6f} N·m")
                print(f"    标准差: {np.std(values):.6f} N·m")
                print(f"    数值分布: min={np.min(values):.6f}, max={np.max(values):.6f}")
        print()
        
        # 显示非零数据详情
        if stats['force_history'] or stats['moment_history']:
            print("🎯 非零数据详情:")
            
            if stats['force_history']:
                print(f"  显著非零合力数据 ({len(stats['force_history'])} 个):")
                for i, entry in enumerate(stats['force_history'][:10]):  # 显示前10个
                    sensor_name = entry['key'].split('.')[-2] if '.' in entry['key'] else 'unknown'
                    print(f"    {i+1:2d}. 样本{entry['index']:3d} {sensor_name}: {entry['force']} (大小: {entry['magnitude']:.6f})")
                if len(stats['force_history']) > 10:
                    print(f"    ... 还有 {len(stats['force_history']) - 10} 个非零合力数据")
            
            if stats['moment_history']:
                print(f"  显著非零合力矩数据 ({len(stats['moment_history'])} 个):")
                for i, entry in enumerate(stats['moment_history'][:10]):  # 显示前10个
                    sensor_name = entry['key'].split('.')[-2] if '.' in entry['key'] else 'unknown'
                    print(f"    {i+1:2d}. 样本{entry['index']:3d} {sensor_name}: {entry['moment']} (大小: {entry['magnitude']:.6f})")
                if len(stats['moment_history']) > 10:
                    print(f"    ... 还有 {len(stats['moment_history']) - 10} 个非零合力矩数据")
        else:
            print("❌ 未发现任何显著的非零触觉数据")
            print("可能的原因:")
            print("  1. 传感器未与物体接触")
            print("  2. 传感器校准问题")
            print("  3. 数据采集配置错误")
            print("  4. 传感器硬件故障")
        
        # 检查时间戳数据
        print("\n🕒 时间戳数据检查:")
        timestamp_keys = [k for k in tac3d_keys if 'timestamp' in k]
        
        for key in timestamp_keys:
            timestamps = []
            for i in range(min(100, len(dataset))):  # 检查前100个样本
                sample = dataset[i]
                ts = sample[key].item() if isinstance(sample[key], torch.Tensor) else sample[key]
                timestamps.append(ts)
            
            timestamps = np.array(timestamps)
            sensor_name = key.split('.')[-2] if '.' in key else 'unknown'
            field_name = key.split('.')[-1] if '.' in key else 'unknown'
            
            print(f"  {sensor_name} {field_name}:")
            print(f"    数值范围: [{np.min(timestamps):.6f}, {np.max(timestamps):.6f}]")
            print(f"    是否全为零: {'是' if np.all(timestamps == 0) else '否'}")
            if not np.all(timestamps == 0):
                print(f"    时间间隔: {np.mean(np.diff(timestamps)):.6f} 秒")
        
        # 检查其他3D数据
        print("\n📊 3D数据数组检查:")
        array_keys = [k for k in tac3d_keys if any(x in k for x in ['positions_3d', 'forces_3d', 'displacements_3d'])]
        
        for key in array_keys:
            field_name = key.split('.')[-1] if '.' in key else 'unknown'
            sensor_name = key.split('.')[-2] if '.' in key else 'unknown'
            
            # 检查前10个样本
            non_zero_count = 0
            max_values = []
            
            for i in range(min(10, len(dataset))):
                sample = dataset[i]
                data = sample[key]
                if torch.any(data != 0):
                    non_zero_count += 1
                max_val = torch.max(torch.abs(data)).item()
                max_values.append(max_val)
            
            print(f"  {sensor_name} {field_name}:")
            print(f"    形状: {sample[key].shape}")
            print(f"    前10个样本中非零样本: {non_zero_count}/10")
            print(f"    最大绝对值: {np.max(max_values):.6f}")
            print(f"    是否全为零: {'是' if np.max(max_values) == 0 else '否'}")
        
        print(f"\n✅ 全面数据检查完成")
        
        # 总结和建议
        print("\n🎯 诊断总结:")
        if stats['non_zero_samples'] == 0:
            print("❌ 所有触觉数据都为零")
            print("🔧 建议检查:")
            print("  1. 传感器连接状态")
            print("  2. 传感器校准是否正确")
            print("  3. 数据采集时是否有物理接触")
            print("  4. 传感器UDP通信是否正常")
        elif stats['non_zero_samples'] < stats['total_samples'] * 0.1:
            print("⚠️  大部分触觉数据为零")
            print("🔧 可能的问题:")
            print("  1. 间歇性接触（正常情况）")
            print("  2. 传感器灵敏度设置")
            print("  3. 部分时间段无接触")
        else:
            print("✅ 触觉数据正常，包含有效的接触信息")
        
    except Exception as e:
        print(f"❌ 检查数据集时出错: {e}")
        import traceback
        traceback.print_exc()

def analyze_episode_patterns(dataset_path: str):
    """分析episode中的数据模式"""
    print(f"\n🎬 Episode数据模式分析...")
    
    try:
        dataset = LeRobotDataset(dataset_path)
        
        for episode_idx in range(min(3, dataset.num_episodes)):  # 最多分析3个episodes
            print(f"\n📊 Episode {episode_idx}:")
            
            from_idx = int(dataset.episode_data_index["from"][episode_idx].item())
            to_idx = int(dataset.episode_data_index["to"][episode_idx].item())
            
            print(f"  帧范围: {from_idx} - {to_idx-1} (共 {to_idx-from_idx} 帧)")
            
            # 分析整个episode的force数据
            force_magnitudes = []
            for i in range(from_idx, to_idx):
                sample = dataset[i]
                for key in sample.keys():
                    if 'resultant_force' in key and 'tactile' in key:
                        force = sample[key]
                        magnitude = torch.norm(force).item()
                        force_magnitudes.append(magnitude)
                        break  # 只取第一个force键
            
            force_magnitudes = np.array(force_magnitudes)
            
            print(f"  合力大小统计:")
            print(f"    平均值: {np.mean(force_magnitudes):.6f}")
            print(f"    最大值: {np.max(force_magnitudes):.6f}")
            print(f"    非零帧数: {np.sum(force_magnitudes > 1e-6)}/{len(force_magnitudes)}")
            
            # 找到力值变化的时间点
            if np.any(force_magnitudes > 1e-6):
                non_zero_indices = np.where(force_magnitudes > 1e-6)[0]
                print(f"    首次非零: 帧 {non_zero_indices[0] + from_idx}")
                print(f"    最后非零: 帧 {non_zero_indices[-1] + from_idx}")
                print(f"    最大力时刻: 帧 {np.argmax(force_magnitudes) + from_idx}")
            else:
                print(f"    ❌ 整个episode无有效力数据")
    
    except Exception as e:
        print(f"❌ Episode分析出错: {e}")

if __name__ == "__main__":
    # 使用完整路径
    dataset_path = "/home/user/.cache/huggingface/lerobot/user/test_two2"
    
    print(f"检查数据集路径: {dataset_path}")
    
    if not Path(dataset_path).exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        exit(1)
    
    # 全面分析数据集
    comprehensive_tac3d_analysis(dataset_path)
    
    # 分析episode模式
    analyze_episode_patterns(dataset_path) 