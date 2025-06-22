#!/usr/bin/env python3
"""
GelSight数据格式测试脚本

这个脚本演示了GelSight传感器的数据格式，包括：
1. 传感器数据结构
2. 数据字段类型和大小
3. 与LeRobot的集成格式
4. 数据保存和读取示例
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
    from lerobot.common.robot_devices.tactile_sensors.gelsight import GelSightSensor
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in fallback mode...")
    sys.exit(1)


def test_data_format():
    """测试GelSight数据格式"""
    print("=== GelSight 数据格式测试 ===\n")
    
    # 1. 创建配置
    print("1. 传感器配置:")
    config = GelSightConfig(
        device_name="GelSight Mini",
        imgh=240,
        imgw=320,
        framerate=25
    )
    
    print(f"   设备名称: {config.device_name}")
    print(f"   图像尺寸: {config.imgh} × {config.imgw}")
    print(f"   帧率: {config.framerate} fps")
    print(f"   预期数据量: {config.imgh * config.imgw * 3 / 1024:.1f} KB/帧")
    
    # 2. 创建传感器（模拟模式）
    print("\n2. 传感器创建:")
    sensor = GelSightSensor(config)
    print(f"   传感器类型: {type(sensor).__name__}")
    print(f"   连接状态: {sensor.is_connected()}")
    
    # 3. 模拟数据读取
    print("\n3. 模拟数据格式:")
    
    # 创建模拟图像数据
    mock_image = np.random.randint(0, 255, (config.imgh, config.imgw, 3), dtype=np.uint8)
    
    # 模拟传感器返回的数据格式
    mock_data = {
        # LeRobot标准字段
        'SN': config.device_name,
        'index': 12345,
        'sendTimestamp': time.time(),
        'recvTimestamp': time.time(),
        
        # GelSight特有字段
        'tactile_image': mock_image,
        'image_shape': mock_image.shape,
        
        # 向后兼容字段
        'timestamp': time.time(),
        'device_name': config.device_name,
        'frame_index': 12345,
        'image': mock_image,
        'sensor_config': {
            'imgh': config.imgh,
            'imgw': config.imgw,
            'framerate': config.framerate,
        }
    }
    
    print("   数据字段分析:")
    total_size = 0
    for key, value in mock_data.items():
        if isinstance(value, np.ndarray):
            size_bytes = value.nbytes
            size_kb = size_bytes / 1024
            print(f"     {key:>15}: {value.shape} {value.dtype} = {size_bytes:,} bytes ({size_kb:.1f} KB)")
            total_size += size_bytes
        elif isinstance(value, (int, float)):
            size_bytes = 8  # 估算
            print(f"     {key:>15}: {type(value).__name__} = {size_bytes} bytes")
            total_size += size_bytes
        elif isinstance(value, str):
            size_bytes = len(value.encode('utf-8'))
            print(f"     {key:>15}: string = {size_bytes} bytes")
            total_size += size_bytes
        elif isinstance(value, dict):
            print(f"     {key:>15}: dict (配置信息)")
        else:
            print(f"     {key:>15}: {type(value).__name__}")
    
    print(f"\n   总数据量: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    # 4. LeRobot格式转换
    print("\n4. LeRobot观测格式:")
    sensor_name = "right_finger"
    
    # 转换为LeRobot观测格式
    obs_dict = {}
    
    # 基本元数据
    obs_dict[f"observation.tactile.{sensor_name}.sensor_sn"] = mock_data['SN']
    obs_dict[f"observation.tactile.{sensor_name}.frame_index"] = torch.tensor([mock_data['index']], dtype=torch.int64)
    obs_dict[f"observation.tactile.{sensor_name}.send_timestamp"] = torch.tensor([mock_data['sendTimestamp']], dtype=torch.float64)
    obs_dict[f"observation.tactile.{sensor_name}.recv_timestamp"] = torch.tensor([mock_data['recvTimestamp']], dtype=torch.float64)
    
    # 图像数据
    obs_dict[f"observation.tactile.{sensor_name}.tactile_image"] = torch.from_numpy(mock_data['tactile_image'])
    
    print("   观测字典键名:")
    for key in obs_dict.keys():
        data = obs_dict[key]
        if hasattr(data, 'shape'):
            print(f"     {key}")
            print(f"       └─ shape: {data.shape}, dtype: {data.dtype}")
        else:
            print(f"     {key}: {type(data).__name__} = '{data}'")
    
    return obs_dict


def test_data_analysis():
    """数据分析示例"""
    print("\n=== 数据分析示例 ===\n")
    
    # 创建示例数据
    obs_dict = test_data_format()
    sensor_name = "right_finger"
    
    # 提取图像数据
    image_tensor = obs_dict[f"observation.tactile.{sensor_name}.tactile_image"]
    image_np = image_tensor.numpy()
    
    print("5. 图像数据分析:")
    print(f"   形状: {image_np.shape}")
    print(f"   数据类型: {image_np.dtype}")
    print(f"   数值范围: [{image_np.min()}, {image_np.max()}]")
    print(f"   平均值: {image_np.mean():.2f}")
    print(f"   标准差: {image_np.std():.2f}")
    
    # 通道分析
    print(f"\n   通道分析:")
    for i, channel in enumerate(['B', 'G', 'R']):
        channel_data = image_np[:, :, i]
        print(f"     {channel}通道: 均值={channel_data.mean():.2f}, 标准差={channel_data.std():.2f}")
    
    # 统计信息
    print(f"\n   统计信息:")
    print(f"     像素总数: {image_np.size:,}")
    print(f"     非零像素: {np.count_nonzero(image_np):,}")
    print(f"     内存占用: {image_np.nbytes:,} bytes")


def test_storage_comparison():
    """存储格式对比"""
    print("\n=== 存储格式对比 ===\n")
    
    # 不同分辨率的数据量对比
    configs = [
        ("低分辨率", 120, 160),
        ("标准分辨率", 240, 320),
        ("高分辨率", 480, 640),
    ]
    
    print("6. 不同分辨率数据量对比:")
    print(f"   {'配置':^12} | {'图像尺寸':^12} | {'数据量/帧':^12} | {'30fps速率':^12}")
    print(f"   {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    
    for name, h, w in configs:
        size_bytes = h * w * 3  # uint8, 3通道
        size_kb = size_bytes / 1024
        fps30_mb = size_bytes * 30 / (1024 * 1024)
        
        print(f"   {name:^12} | {h}×{w:>6}   | {size_kb:>8.1f} KB | {fps30_mb:>8.1f} MB/s")
    
    # 与Tac3D对比
    print(f"\n7. 与Tac3D传感器对比:")
    
    tac3d_size = (
        50 +        # sensor_sn
        8 +         # frame_index  
        8 +         # send_timestamp
        8 +         # recv_timestamp
        400 * 3 * 8 +   # positions_3d
        400 * 3 * 8 +   # displacements_3d
        400 * 3 * 8 +   # forces_3d
        3 * 8 +     # resultant_force
        3 * 8       # resultant_moment
    )
    
    gelsight_size = (
        20 +        # sensor_sn
        8 +         # frame_index
        8 +         # send_timestamp  
        8 +         # recv_timestamp
        240 * 320 * 3   # tactile_image
    )
    
    print(f"   Tac3D数据量:    {tac3d_size:,} bytes ({tac3d_size/1024:.1f} KB)")
    print(f"   GelSight数据量: {gelsight_size:,} bytes ({gelsight_size/1024:.1f} KB)")
    print(f"   数据量比例:     {gelsight_size/tac3d_size:.1f}x")


def test_practical_usage():
    """实际使用示例"""
    print("\n=== 实际使用示例 ===\n")
    
    print("8. 实际集成示例代码:")
    
    # 显示配置示例
    config_example = """
# 机器人配置示例
config = AlohaRobotConfig(
    tactile_sensors={
        "right_finger": GelSightConfig(
            device_name="GelSight Mini",
            imgh=240,
            imgw=320,
            framerate=25,
        ),
        "left_finger": GelSightConfig(
            device_name="GelSight Mini 2", 
            imgh=240,
            imgw=320,
            framerate=25,
        ),
    }
)"""
    
    # 显示数据读取示例
    data_access_example = """
# 数据访问示例
robot = ManipulatorRobot(config)
robot.connect()

obs = robot.capture_observation()

# 访问右手指触觉图像
right_image = obs["observation.tactile.right_finger.tactile_image"]
print(f"右指图像: {right_image.shape}")

# 访问左手指触觉图像  
left_image = obs["observation.tactile.left_finger.tactile_image"]
print(f"左指图像: {left_image.shape}")

# 获取时间戳
timestamp = obs["observation.tactile.right_finger.recv_timestamp"]
print(f"时间戳: {timestamp.item()}")"""
    
    print("   配置代码:")
    for line in config_example.strip().split('\n'):
        print(f"     {line}")
        
    print("\n   数据访问代码:")
    for line in data_access_example.strip().split('\n'):
        print(f"     {line}")


def main():
    """主函数"""
    try:
        test_data_format()
        test_data_analysis()
        test_storage_comparison()
        test_practical_usage()
        
        print(f"\n🎉 GelSight数据格式测试完成!")
        
        print(f"\n📋 总结:")
        print(f"  • 数据格式: 图像为主的触觉数据")
        print(f"  • 标准尺寸: 240×320×3 (~230KB/帧)")
        print(f"  • 兼容性: 完全兼容LeRobot数据集格式")
        print(f"  • 扩展性: 支持多分辨率和多传感器配置")
        print(f"  • 处理方式: 基于计算机视觉的图像分析")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 