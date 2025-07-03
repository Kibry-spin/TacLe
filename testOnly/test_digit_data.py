#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本：检查现有数据集中的DIGIT传感器数据格式
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 尝试导入lerobot相关模块
try:
    from lerobot.common.datasets.v2.dataset import Dataset
except ImportError:
    print("无法导入lerobot模块，请确保环境正确配置")
    sys.exit(1)

def check_digit_data(dataset_path):
    """检查数据集中的DIGIT传感器数据格式"""
    print(f"正在检查数据集: {dataset_path}")
    
    try:
        # 尝试加载数据集
        dataset = Dataset(dataset_path)
        print(f"数据集加载成功，共有 {len(dataset)} 帧")
        
        # 检查数据集特征
        features = dataset.features
        print("\n数据集特征:")
        
        # 查找DIGIT相关特征
        digit_features = {}
        for key, feature in features.items():
            if 'tactile' in key and 'digit' in key:
                digit_features[key] = feature
                print(f"  {key}: {feature}")
        
        if not digit_features:
            print("数据集中未找到DIGIT传感器数据")
            return
        
        # 检查第一帧数据
        print("\n检查第一帧DIGIT数据:")
        frame_0 = dataset[0]
        
        for key in digit_features.keys():
            if key in frame_0:
                data = frame_0[key]
                print(f"  {key}:")
                print(f"    类型: {type(data)}")
                
                if isinstance(data, (torch.Tensor, np.ndarray)):
                    print(f"    形状: {data.shape}")
                    print(f"    数据类型: {data.dtype}")
                    
                    # 对于图像数据，检查数值范围
                    if 'tactile_image' in key:
                        try:
                            if isinstance(data, torch.Tensor):
                                min_val = data.min().item()
                                max_val = data.max().item()
                            else:
                                min_val = data.min()
                                max_val = data.max()
                            print(f"    数值范围: [{min_val}, {max_val}]")
                            
                            # 显示图像
                            plt.figure(figsize=(8, 6))
                            if isinstance(data, torch.Tensor):
                                img_data = data.numpy() if data.ndim == 3 else data.reshape(240, 320, 3).numpy()
                            else:
                                img_data = data if data.ndim == 3 else data.reshape(240, 320, 3)
                            
                            plt.imshow(img_data)
                            plt.title(f"DIGIT传感器图像 - {key}")
                            plt.colorbar()
                            plt.savefig(f"digit_image_{Path(key).name}.png")
                            print(f"    图像已保存为: digit_image_{Path(key).name}.png")
                        except Exception as e:
                            print(f"    无法处理图像数据: {e}")
                else:
                    print(f"    值: {data}")
            else:
                print(f"  {key}: 未找到")
        
        # 检查多帧数据
        print("\n检查多帧数据的一致性:")
        sample_indices = [0, min(10, len(dataset)-1), min(100, len(dataset)-1)]
        for idx in sample_indices:
            if idx >= len(dataset):
                continue
                
            frame = dataset[idx]
            print(f"\n帧 {idx}:")
            
            for key in digit_features.keys():
                if key in frame and 'tactile_image' in key:
                    data = frame[key]
                    if isinstance(data, (torch.Tensor, np.ndarray)):
                        print(f"  {key}: 形状={data.shape}, 类型={data.dtype}")
                    else:
                        print(f"  {key}: 类型={type(data)}")
    
    except Exception as e:
        print(f"检查数据集时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # 默认数据集路径
        dataset_path = input("请输入数据集路径: ")
        if not dataset_path:
            print("未提供数据集路径，退出程序")
            return
    
    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return
    
    # 检查数据集
    check_digit_data(dataset_path)

if __name__ == "__main__":
    main() 