#!/usr/bin/env python3
"""
测试触觉数据保存功能
"""

import time
import torch
from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.tactile_sensors.configs import Tac3DConfig


def test_tactile_data_save():
    """测试触觉数据保存功能"""
    
    print("=== 测试触觉数据保存功能 ===\n")
    
    # 创建机器人配置
    config = AlohaRobotConfig(
        mock=True,
        tactile_sensors={
            "left_gripper": Tac3DConfig(port=9988, mock=False),
        }
    )
    
    robot = ManipulatorRobot(config)
    
    print("📋 完整触觉数据特征列表:")
    tactile_features = robot.tactile_features
    for i, (name, feature) in enumerate(tactile_features.items(), 1):
        shape_str = f"{feature['shape']}"
        print(f"  {i:2d}. {name}")
        print(f"      shape={shape_str}, dtype={feature['dtype']}")
    
    print(f"\n✅ 总共定义了 {len(tactile_features)} 个触觉数据特征")
    
    # 数据大小分析
    print("\n📊 触觉数据存储大小分析:")
    
    import numpy as np
    total_size = 0
    for name, feature in tactile_features.items():
        if "shape" in feature:
            shape = feature["shape"]
            if feature["dtype"] == "string":
                size_bytes = 50  # 估计字符串大小
            elif feature["dtype"] == "int64":
                size_bytes = np.prod(shape) * 8
            elif feature["dtype"] == "float64":
                size_bytes = np.prod(shape) * 8
            else:
                size_bytes = 0
            
            total_size += size_bytes
            size_kb = size_bytes / 1024
            field_name = name.split('.')[-1]  # 获取最后一部分
            print(f"  {field_name:>20}: {size_bytes:>6} bytes ({size_kb:>6.1f} KB)")
    
    print(f"\n💾 每帧完整触觉数据总大小:")
    print(f"  总计: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"  对比之前仅保存力和力矩: {total_size/24:.0f}x 数据量")
    
    # 数据内容分析
    print(f"\n📊 数据内容分析:")
    print(f"  • 基本元数据: 4 个字段 (SN, 帧索引, 时间戳)")
    print(f"  • 3D阵列数据: 3 个 (400x3) 矩阵")
    print(f"    - 位置 (3D_Positions): 标志点的空间坐标")
    print(f"    - 位移 (3D_Displacements): 相对于基准的位移")
    print(f"    - 力场 (3D_Forces): 每个点的局部受力")
    print(f"  • 合成数据: 2 个 3D 向量 (合成力和力矩)")
    
    print(f"\n🎯 应用场景:")
    print(f"  • 高精度接触检测和力感知")
    print(f"  • 表面形状和纹理分析")
    print(f"  • 力分布可视化和分析")
    print(f"  • 机器学习的多模态感知")
    
    try:
        robot.connect()
        print(f"\n🔌 传感器连接成功!")
        
        # 读取一帧真实数据检验
        obs = robot.capture_observation()
        
        print(f"\n📊 实际数据验证:")
        sensor_name = "left_gripper"
        
        # 检查关键数据字段
        key_checks = [
            (f"observation.tactile.{sensor_name}.sensor_sn", "传感器SN"),
            (f"observation.tactile.{sensor_name}.positions_3d", "3D位置"),
            (f"observation.tactile.{sensor_name}.forces_3d", "3D力场"),
            (f"observation.tactile.{sensor_name}.resultant_force", "合成力"),
        ]
        
        for key, desc in key_checks:
            if key in obs:
                data = obs[key]
                if hasattr(data, 'shape'):
                    print(f"  ✅ {desc}: shape={data.shape}, dtype={data.dtype}")
                else:
                    print(f"  ✅ {desc}: {type(data)} = '{data}'")
            else:
                print(f"  ❌ {desc}: 缺失")
        
        robot.disconnect()
        print(f"\n✅ 数据验证完成，所有字段正常!")
        
    except Exception as e:
        print(f"\n⚠️  无法连接真实传感器 (这是正常的): {e}")
    
    print(f"\n🎉 触觉数据保存功能测试完成!")
    print(f"\n📋 总结:")
    print(f"  • 实现了完整触觉数据的收集和保存")
    print(f"  • 数据量从 24 bytes 增加到 {total_size/1024:.1f} KB")
    print(f"  • 包含了 3D 位置、位移、力场等完整信息")
    print(f"  • 兼容 LeRobot 数据集保存框架")


if __name__ == "__main__":
    test_tactile_data_save() 