#!/usr/bin/env python3
"""
测试GelSight传感器的资源清理功能
特别是在KeyboardInterrupt情况下的资源释放
"""

import time
import signal
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
from lerobot.common.robot_devices.tactile_sensors.gelsight import GelSightSensor


def test_resource_cleanup_on_interrupt():
    """测试在KeyboardInterrupt时的资源清理"""
    print("=== GelSight 资源清理测试 ===")
    
    # 创建传感器配置
    config = GelSightConfig(device_name="GelSight Mini")
    sensor = GelSightSensor(config)
    
    try:
        # 连接传感器
        print("连接GelSight传感器...")
        sensor.connect()
        print(f"传感器连接成功: {sensor.get_sensor_info()}")
        
        # 模拟数据读取
        print("开始读取数据，5秒后会模拟KeyboardInterrupt...")
        print("你也可以手动按Ctrl+C测试中断处理")
        
        start_time = time.time()
        frame_count = 0
        
        # 注册信号处理器
        def signal_handler(signum, frame):
            print(f"\n收到信号 {signum}，开始清理资源...")
            try:
                sensor.disconnect()
                print("传感器资源清理完成")
            except Exception as e:
                print(f"清理过程中出错: {e}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        while time.time() - start_time < 5:
            try:
                data = sensor.read()
                if data and 'image' in data:
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"已读取 {frame_count} 帧")
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n收到KeyboardInterrupt...")
                break
        
        # 如果没有被中断，自动模拟中断
        if time.time() - start_time >= 5:
            print("\n模拟KeyboardInterrupt...")
            raise KeyboardInterrupt()
            
    except KeyboardInterrupt:
        print("处理KeyboardInterrupt...")
        # 传感器的read方法应该已经处理了清理
    except Exception as e:
        print(f"测试过程中出错: {e}")
    finally:
        # 确保资源被释放
        try:
            if sensor.is_connected():
                print("执行最终清理...")
                sensor.disconnect()
        except:
            pass
        print("测试完成")


def test_multiple_connect_disconnect():
    """测试多次连接断开是否会导致资源泄漏"""
    print("\n=== 多次连接断开测试 ===")
    
    config = GelSightConfig(device_name="GelSight Mini")
    
    for i in range(3):
        print(f"\n第 {i+1} 次连接测试:")
        sensor = GelSightSensor(config)
        
        try:
            print("  连接传感器...")
            sensor.connect()
            print("  连接成功")
            
            # 读取几帧数据
            for j in range(5):
                data = sensor.read()
                if data and 'image' in data:
                    print(f"    读取第 {j+1} 帧: 形状 {data['image'].shape}")
                time.sleep(0.1)
            
            print("  断开连接...")
            sensor.disconnect()
            print("  断开成功")
            
        except Exception as e:
            print(f"  测试失败: {e}")
            try:
                sensor.disconnect()
            except:
                pass
        
        # 等待一下确保资源完全释放
        time.sleep(1)
    
    print("多次连接断开测试完成")


def test_force_cleanup():
    """测试强制清理功能"""
    print("\n=== 强制清理测试 ===")
    
    config = GelSightConfig(device_name="GelSight Mini")
    sensor = GelSightSensor(config)
    
    try:
        print("连接传感器...")
        sensor.connect()
        
        print("读取一些数据...")
        for i in range(10):
            data = sensor.read()
            if data and 'image' in data:
                print(f"读取第 {i+1} 帧")
            time.sleep(0.1)
        
        print("强制设置连接状态为False（模拟异常情况）...")
        # 不调用正常disconnect，而是强制清理
        if hasattr(sensor, '_device') and sensor._device:
            print("直接调用设备的release方法...")
            sensor._device.release()
            sensor._device = None
            sensor._connected = False
        
        print("强制清理完成")
        
    except Exception as e:
        print(f"强制清理测试出错: {e}")


if __name__ == "__main__":
    try:
        # 测试1: KeyboardInterrupt处理
        test_resource_cleanup_on_interrupt()
        
        # 测试2: 多次连接断开
        test_multiple_connect_disconnect()
        
        # 测试3: 强制清理
        test_force_cleanup()
        
        print("\n所有测试完成!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    finally:
        print("脚本结束") 