#!/usr/bin/env python3
"""
简化的GelSight机器人集成测试脚本

本脚本专注于测试GelSight传感器在LeRobot配置系统中的集成，
不需要实际的硬件连接。

主要测试：
1. 配置系统集成
2. 传感器实例创建
3. 特征定义
4. 数据格式
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

def test_config_imports():
    """测试配置导入"""
    print("=== 测试配置导入 ===")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
        print("✓ GelSightConfig 导入成功")
        
        from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig
        print("✓ AlohaRobotConfig 导入成功")
        
        # 测试配置创建
        gelsight_config = GelSightConfig(
            device_name="Test GelSight",
            imgh=240,
            imgw=320,
            framerate=25,
            mock=True
        )
        print(f"✓ GelSight配置创建成功: {gelsight_config.type}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置导入失败: {e}")
        return False


def test_sensor_factory():
    """测试传感器工厂方法"""
    print("\n=== 测试传感器工厂 ===")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.utils import make_tactile_sensor
        from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
        
        # 测试GelSight传感器创建
        sensor = make_tactile_sensor(
            "gelsight",
            device_name="Test GelSight",
            imgh=240,
            imgw=320,
            mock=True
        )
        print("✓ GelSight传感器实例创建成功")
        print(f"  设备名称: {sensor.device_name}")
        print(f"  连接状态: {sensor.is_connected()}")
        
        # 测试传感器信息
        info = sensor.get_sensor_info()
        print(f"  传感器信息: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ 传感器工厂测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_robot_config_with_gelsight():
    """测试机器人配置中的GelSight集成"""
    print("\n=== 测试机器人配置集成 ===")
    
    try:
        from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig
        from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
        
        # 创建包含GelSight的Aloha配置
        config = AlohaRobotConfig()
        config.mock = True
        
        # 替换为GelSight传感器
        config.tactile_sensors = {
            "left_gripper": GelSightConfig(
                device_name="GelSight Left",
                imgh=240,
                imgw=320,
                framerate=30,
                mock=True,
            ),
            "right_gripper": GelSightConfig(
                device_name="GelSight Right",
                imgh=240,
                imgw=320,
                framerate=30,
                mock=True,
            ),
        }
        
        print(f"✓ Aloha配置创建成功")
        print(f"  触觉传感器数量: {len(config.tactile_sensors)}")
        
        for name, sensor_config in config.tactile_sensors.items():
            print(f"    {name}: {sensor_config.type}")
            print(f"      设备: {sensor_config.device_name}")
            print(f"      尺寸: {sensor_config.imgh}×{sensor_config.imgw}")
        
        # 将config存储为全局变量以供其他测试使用
        global test_config
        test_config = config
        
        return True
        
    except Exception as e:
        print(f"❌ 机器人配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tactile_features():
    """测试触觉特征定义"""
    print("\n=== 测试触觉特征定义 ===")
    
    try:
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
        
        # 使用全局变量中的配置
        if 'test_config' not in globals():
            print("❌ 测试配置不可用")
            return False
        
        config = test_config
        
        # 创建机器人实例（不连接）
        robot = ManipulatorRobot(config)
        print("✓ 机器人实例创建成功")
        
        # 测试特征定义
        features = robot.features
        print(f"✓ 总特征数: {len(features)}")
        
        tactile_features = robot.tactile_features
        print(f"✓ 触觉特征数: {len(tactile_features)}")
        
        # 详细显示触觉特征
        print("\n  触觉特征详情:")
        for key, feature in tactile_features.items():
            if "tactile_image" in key:
                shape = feature['shape']
                dtype = feature['dtype']
                names = feature.get('names', [])
                print(f"    {key}:")
                print(f"      形状: {shape}")
                print(f"      类型: {dtype}")
                print(f"      维度名: {names}")
        
        return True
        
    except Exception as e:
        print(f"❌ 触觉特征测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sensor_creation_from_config():
    """测试从配置创建传感器"""
    print("\n=== 测试从配置创建传感器 ===")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.utils import make_tactile_sensors_from_configs
        from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
        
        # 创建传感器配置字典
        tactile_configs = {
            "left_finger": GelSightConfig(
                device_name="GelSight Left",
                imgh=240,
                imgw=320,
                mock=True
            ),
            "right_finger": GelSightConfig(
                device_name="GelSight Right",
                imgh=120,
                imgw=160,
                mock=True
            ),
        }
        
        # 从配置创建传感器
        sensors = make_tactile_sensors_from_configs(tactile_configs)
        print(f"✓ 从配置创建了 {len(sensors)} 个传感器")
        
        for name, sensor in sensors.items():
            print(f"    {name}:")
            print(f"      设备名: {sensor.device_name}")
            print(f"      图像尺寸: {sensor.imgh}×{sensor.imgw}")
            print(f"      连接状态: {sensor.is_connected()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 传感器创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_standardization():
    """测试数据标准化"""
    print("\n=== 测试数据标准化 ===")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.utils import standardize_tactile_data
        import numpy as np
        
        # 模拟GelSight数据
        raw_data = {
            'tactile_image': np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            'device_name': 'Test GelSight',
            'frame_index': 42,
            'timestamp': 1234567890.123
        }
        
        standardized = standardize_tactile_data(
            raw_data=raw_data,
            sensor_type="gelsight",
            sensor_id="test_sensor"
        )
        
        print("✓ 数据标准化成功")
        print(f"  标准化字段:")
        for key, value in standardized.items():
            if key == 'tactile_image':
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"    {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据标准化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🔧 GelSight机器人集成简化测试")
    print("=" * 50)
    
    tests = [
        test_config_imports,
        test_sensor_factory,
        test_robot_config_with_gelsight,
        test_tactile_features,
        test_sensor_creation_from_config,
        test_standardization,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 异常: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试总结:")
    passed = sum(results)
    total = len(results)
    
    print(f"  通过: {passed}/{total}")
    print(f"  成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！GelSight已成功集成到LeRobot中。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 