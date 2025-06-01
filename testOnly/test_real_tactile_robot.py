#!/usr/bin/env python3
"""
测试真实触觉传感器与机器人集成
"""

import time
import numpy as np
from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.tactile_sensors.configs import Tac3DConfig


def test_real_tactile_robot():
    """测试真实触觉传感器与机器人的集成"""
    
    print("=== 测试真实触觉传感器与机器人集成 ===\n")
    
    # 创建机器人配置，使用真实传感器
    config = AlohaRobotConfig(
        mock=True,  # 机器人其他部分使用mock，只测试触觉传感器
        tactile_sensors={
            "left_gripper": Tac3DConfig(
                port=9988,  # 确保这是你的真实传感器端口
                auto_calibrate=True,
                mock=False,  # 使用真实传感器！
            ),
        }
    )
    
    print(f"配置的触觉传感器: {list(config.tactile_sensors.keys())}")
    print(f"传感器端口: {config.tactile_sensors['left_gripper'].port}")
    print(f"是否mock: {config.tactile_sensors['left_gripper'].mock}")
    
    # 创建机器人实例
    robot = ManipulatorRobot(config)
    print(f"\n机器人创建成功，包含触觉传感器: {list(robot.tactile_sensors.keys())}")
    
    try:
        # 连接机器人（包括触觉传感器）
        print("\n🔌 正在连接机器人和触觉传感器...")
        robot.connect()
        print("✅ 连接成功!")
        
        # 检查传感器连接状态
        for name, sensor in robot.tactile_sensors.items():
            print(f"  {name}: 连接状态 = {sensor.is_connected()}")
            sensor_info = sensor.get_sensor_info()
            if 'sensor_sn' in sensor_info:
                print(f"    传感器SN: {sensor_info['sensor_sn']}")
        
        # 测试数据读取
        print("\n📊 开始读取触觉数据...")
        for i in range(10):
            print(f"\n--- 读取 #{i+1} ---")
            
            # 通过机器人捕获观测数据（包含触觉数据）
            start_time = time.time()
            obs = robot.capture_observation()
            read_time = time.time() - start_time
            
            print(f"数据读取耗时: {read_time*1000:.1f}ms")
            
            # 显示触觉数据
            for sensor_name in robot.tactile_sensors.keys():
                force_key = f"observation.tactile.{sensor_name}.force"
                moment_key = f"observation.tactile.{sensor_name}.moment"
                
                if force_key in obs and moment_key in obs:
                    force = obs[force_key].numpy()
                    moment = obs[moment_key].numpy()
                    
                    force_magnitude = np.linalg.norm(force)
                    moment_magnitude = np.linalg.norm(moment)
                    
                    print(f"  {sensor_name}:")
                    print(f"    Force:  [{force[0]:7.3f}, {force[1]:7.3f}, {force[2]:7.3f}] |F|={force_magnitude:7.3f}")
                    print(f"    Moment: [{moment[0]:7.3f}, {moment[1]:7.3f}, {moment[2]:7.3f}] |M|={moment_magnitude:7.3f}")
                else:
                    print(f"  {sensor_name}: ❌ 数据不可用")
            
            time.sleep(0.5)
        
        print("\n✅ 触觉传感器数据读取测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 断开连接
        if robot.is_connected:
            print("\n🔌 断开机器人连接...")
            robot.disconnect()
            print("✅ 断开完成")


def test_sensor_discovery():
    """测试传感器发现功能"""
    print("\n=== 测试传感器发现 ===")
    
    from lerobot.common.robot_devices.tactile_sensors.tac3d import Tac3DSensor
    from lerobot.common.robot_devices.tactile_sensors.configs import Tac3DConfig
    
    # 测试常用端口
    test_ports = [9988]
    
    for port in test_ports:
        print(f"\n测试端口 {port}...")
        config = Tac3DConfig(port=port, auto_calibrate=False, mock=False)
        sensor = Tac3DSensor(config)
        
        try:
            sensor.connect()
            if sensor.is_connected():
                info = sensor.get_sensor_info()
                print(f"  ✅ 找到传感器: {info}")
                sensor.disconnect()
            else:
                print(f"  ❌ 端口 {port} 无传感器")
        except Exception as e:
            print(f"  ❌ 端口 {port} 连接失败: {e}")


if __name__ == "__main__":
    # 首先测试传感器发现
    test_sensor_discovery()
    
    print("\n" + "="*60 + "\n")
    
    # 然后测试机器人集成
    test_real_tactile_robot()
    
    print("\n🎉 测试完成!") 