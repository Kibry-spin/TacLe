#!/usr/bin/env python3
"""
GelSight机器人集成测试脚本

本脚本展示如何在LeRobot机器人配置中集成GelSight触觉传感器，包括：
1. 创建包含GelSight传感器的机器人配置
2. 初始化机器人和传感器
3. 读取触觉数据
4. 数据格式验证
5. 实际应用示例

使用方法：
python test_robot_gelsight_integration.py [--config CONFIG_TYPE] [--mock] [--duration SECONDS]

示例：
python test_robot_gelsight_integration.py --config aloha --mock --duration 10
python test_robot_gelsight_integration.py --config koch --duration 5
"""

import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from lerobot.common.robot_devices.robots.configs import (
        ManipulatorRobotConfig,
        AlohaRobotConfig,
        KochRobotConfig,
    )
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
    from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
    from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
    
    print("✓ 成功导入LeRobot模块")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在正确的LeRobot环境中运行此脚本")
    sys.exit(1)


@dataclass 
class TestRobotConfig(ManipulatorRobotConfig):
    """
    用于测试GelSight集成的自定义机器人配置
    """
    
    # 基本配置
    calibration_dir: str = ".cache/calibration/test_robot"
    max_relative_target: int | None = None
    mock: bool = True
    
    # 简单的电机配置（用于测试）
    leader_arms: Dict[str, DynamixelMotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/ttyUSB0",  # 模拟端口
                motors={
                    "joint1": [1, "xl330-m077"],
                    "joint2": [2, "xl330-m077"], 
                    "gripper": [3, "xl330-m077"],
                },
                mock=True,
            ),
        }
    )
    
    follower_arms: Dict[str, DynamixelMotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/ttyUSB1",  # 模拟端口
                motors={
                    "joint1": [1, "xl430-w250"],
                    "joint2": [2, "xl430-w250"],
                    "gripper": [3, "xl430-w250"],
                },
                mock=True,
            ),
        }
    )
    
    # 相机配置（可选）
    cameras: Dict[str, OpenCVCameraConfig] = field(
        default_factory=lambda: {
            "wrist_cam": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640, 
                height=480,
                mock=True,
            ),
        }
    )
    
    # GelSight触觉传感器配置
    tactile_sensors: Dict[str, GelSightConfig] = field(
        default_factory=lambda: {
            "right_finger": GelSightConfig(
                device_name="GelSight Mini Right",
                imgh=240,
                imgw=320,
                framerate=25,
                mock=True,  # 模拟模式
            ),
            "left_finger": GelSightConfig(
                device_name="GelSight Mini Left", 
                imgh=240,
                imgw=320,
                framerate=25,
                mock=True,  # 模拟模式
            ),
        }
    )


def create_aloha_with_gelsight(mock: bool = True) -> AlohaRobotConfig:
    """创建包含GelSight传感器的Aloha机器人配置"""
    
    # 使用原有的Aloha配置
    config = AlohaRobotConfig()
    config.mock = mock
    
    # 添加GelSight传感器到夹爪
    config.tactile_sensors = {
        "left_gripper": GelSightConfig(
            device_name="GelSight Left Gripper",
            imgh=240,
            imgw=320, 
            framerate=30,
            mock=mock,
        ),
        "right_gripper": GelSightConfig(
            device_name="GelSight Right Gripper",
            imgh=240,
            imgw=320,
            framerate=30, 
            mock=mock,
        ),
    }
    
    return config


def create_koch_with_gelsight(mock: bool = True) -> KochRobotConfig:
    """创建包含GelSight传感器的Koch机器人配置"""
    
    config = KochRobotConfig()
    config.mock = mock
    
    # 为Koch机器人添加GelSight传感器
    config.tactile_sensors = {
        "gripper_tip": GelSightConfig(
            device_name="GelSight Koch Gripper",
            imgh=240,
            imgw=320,
            framerate=25,
            mock=mock,
        ),
    }
    
    return config


def test_robot_configuration():
    """测试机器人配置创建"""
    print("\n=== 机器人配置测试 ===")
    
    configs = {
        "test": TestRobotConfig(),
        "aloha": create_aloha_with_gelsight(mock=True),
        "koch": create_koch_with_gelsight(mock=True),
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper()} 机器人配置:")
        print(f"  触觉传感器数量: {len(config.tactile_sensors)}")
        
        for sensor_name, sensor_config in config.tactile_sensors.items():
            print(f"    {sensor_name}:")
            print(f"      设备名称: {sensor_config.device_name}")
            print(f"      图像尺寸: {sensor_config.imgh}×{sensor_config.imgw}")
            print(f"      帧率: {sensor_config.framerate}")
            print(f"      模拟模式: {sensor_config.mock}")
            print(f"      传感器类型: {sensor_config.type}")
    
    return configs


def test_robot_initialization(config: ManipulatorRobotConfig):
    """测试机器人初始化"""
    print(f"\n=== 机器人初始化测试 ===")
    
    try:
        # 创建机器人实例
        robot = ManipulatorRobot(config)
        print(f"✓ 机器人实例创建成功")
        
        # 检查特征
        features = robot.features
        print(f"✓ 机器人特征提取成功")
        print(f"  总特征数: {len(features)}")
        
        # 检查触觉特征
        tactile_features = robot.tactile_features
        print(f"  触觉特征数: {len(tactile_features)}")
        
        for key, feature in tactile_features.items():
            if "tactile_image" in key:
                shape = feature['shape']
                dtype = feature['dtype']
                print(f"    {key}: {shape} ({dtype})")
        
        # 连接机器人
        print(f"\n连接机器人...")
        robot.connect()
        print(f"✓ 机器人连接成功")
        print(f"  连接状态: {robot.is_connected}")
        
        return robot
        
    except Exception as e:
        print(f"❌ 机器人初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tactile_data_reading(robot: ManipulatorRobot, duration: float = 5.0):
    """测试触觉数据读取"""
    print(f"\n=== 触觉数据读取测试 ===")
    
    if not robot.is_connected:
        print("❌ 机器人未连接")
        return
    
    print(f"开始读取触觉数据，持续 {duration} 秒...")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            # 捕获观测数据
            obs = robot.capture_observation()
            frame_count += 1
            
            # 分析触觉数据
            tactile_data_found = False
            for key, value in obs.items():
                if "tactile" in key and "tactile_image" in key:
                    tactile_data_found = True
                    sensor_name = key.split('.')[2]  # 提取传感器名称
                    
                    if frame_count % 10 == 1:  # 每10帧显示一次
                        if isinstance(value, torch.Tensor):
                            shape = value.shape
                            dtype = value.dtype
                            print(f"  {sensor_name}: {shape} {dtype}")
                            
                            # 检查数据有效性
                            if len(shape) == 3 and shape[2] == 3:  # H×W×3
                                print(f"    ✓ 图像数据格式正确")
                                print(f"    数值范围: [{value.min()}, {value.max()}]")
                            else:
                                print(f"    ❌ 图像数据格式错误")
            
            if not tactile_data_found and frame_count == 1:
                print("❌ 未找到触觉图像数据")
                break
                
            time.sleep(0.1)  # 100ms间隔
    
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"❌ 数据读取错误: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"\n测试完成:")
    print(f"  读取帧数: {frame_count}")
    print(f"  用时: {elapsed_time:.1f} 秒") 
    print(f"  平均帧率: {fps:.1f} fps")


def test_data_format_validation(robot: ManipulatorRobot):
    """测试数据格式验证"""
    print(f"\n=== 数据格式验证测试 ===")
    
    if not robot.is_connected:
        print("❌ 机器人未连接")
        return
    
    try:
        obs = robot.capture_observation()
        
        # 寻找触觉传感器数据
        tactile_sensors = []
        for key in obs.keys():
            if "tactile" in key and "sensor_sn" in key:
                sensor_name = key.split('.')[2]
                if sensor_name not in tactile_sensors:
                    tactile_sensors.append(sensor_name)
        
        print(f"发现 {len(tactile_sensors)} 个触觉传感器:")
        
        for sensor_name in tactile_sensors:
            print(f"\n  传感器: {sensor_name}")
            
            # 检查必需字段
            required_fields = [
                "sensor_sn", "frame_index",
                "send_timestamp", "recv_timestamp", 
                "tactile_image"
            ]
            
            all_fields_present = True
            for field in required_fields:
                key = f"observation.tactile.{sensor_name}.{field}"
                if key in obs:
                    value = obs[key]
                    if field == "tactile_image":
                        if isinstance(value, torch.Tensor) and len(value.shape) == 3:
                            print(f"    ✓ {field}: {value.shape} {value.dtype}")
                        else:
                            print(f"    ❌ {field}: 格式错误")
                            all_fields_present = False
                    else:
                        print(f"    ✓ {field}: {type(value).__name__}")
                else:
                    print(f"    ❌ {field}: 缺失")
                    all_fields_present = False
            
            if all_fields_present:
                print(f"    ✅ 传感器 {sensor_name} 数据格式正确")
            else:
                print(f"    ❌ 传感器 {sensor_name} 数据格式有问题")
                
    except Exception as e:
        print(f"❌ 数据格式验证失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GelSight机器人集成测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python test_robot_gelsight_integration.py --config test --mock --duration 10
  python test_robot_gelsight_integration.py --config aloha --mock --duration 5
  python test_robot_gelsight_integration.py --config koch --duration 3
        """
    )
    
    parser.add_argument(
        "--config",
        choices=["test", "aloha", "koch"],
        default="test",
        help="机器人配置类型 (default: test)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="启用模拟模式"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="数据读取测试持续时间(秒) (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    print("🤖 GelSight机器人集成测试")
    print("=" * 50)
    
    try:
        # 1. 配置测试
        configs = test_robot_configuration()
        
        # 2. 选择配置
        if args.config == "test":
            config = TestRobotConfig()
        elif args.config == "aloha":
            config = create_aloha_with_gelsight(mock=args.mock)
        elif args.config == "koch":
            config = create_koch_with_gelsight(mock=args.mock)
        
        config.mock = args.mock
        
        print(f"\n使用配置: {args.config.upper()}")
        print(f"模拟模式: {config.mock}")
        
        # 3. 机器人初始化测试
        robot = test_robot_initialization(config)
        if robot is None:
            return 1
        
        try:
            # 4. 数据格式验证测试
            test_data_format_validation(robot)
            
            # 5. 数据读取测试
            test_tactile_data_reading(robot, args.duration)
            
        finally:
            # 6. 清理
            print(f"\n=== 清理资源 ===")
            robot.disconnect()
            print(f"✓ 机器人已断开连接")
        
        print(f"\n🎉 所有测试完成!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断测试")
        return 1
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 