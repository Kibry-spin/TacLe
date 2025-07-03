#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本：测试DIGIT传感器的数据采集和保存
"""

import os
import sys
import numpy as np
import cv2
import time
from pathlib import Path
import threading
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from lerobot.common.robot_devices.tactile_sensors.configs import DIGITConfig
    from lerobot.common.robot_devices.tactile_sensors.digit import DIGITSensor
except ImportError:
    print("无法导入lerobot模块，请确保环境正确配置")
    sys.exit(1)

class DIGITTester:
    """DIGIT传感器测试类"""
    
    def __init__(self, device_name="DIGIT", save_dir="digit_test_output"):
        self.device_name = device_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建传感器配置和实例
        self.config = DIGITConfig(device_name=device_name)
        self.sensor = DIGITSensor(self.config)
        
        # 状态变量
        self.is_running = False
        self.frame_count = 0
        self.start_time = 0
        self.saved_frames = []
        
        # 线程锁
        self.lock = threading.Lock()
    
    def connect(self):
        """连接传感器"""
        try:
            self.sensor.connect()
            print(f"成功连接到DIGIT传感器: {self.device_name}")
            print(f"传感器信息: {self.sensor.get_sensor_info()}")
            return True
        except Exception as e:
            print(f"连接DIGIT传感器失败: {e}")
            return False
    
    def disconnect(self):
        """断开传感器连接"""
        try:
            self.sensor.disconnect()
            print(f"已断开DIGIT传感器: {self.device_name}")
        except:
            pass
    
    def capture_frame(self):
        """采集一帧数据并返回"""
        try:
            data = self.sensor.read()
            if data and 'tactile_image' in data:
                image = data['tactile_image']
                if image is not None:
                    # 检查图像格式
                    if isinstance(image, np.ndarray):
                        if image.ndim == 3 and image.shape[2] == 3:
                            # 图像格式正确
                            return data
                        else:
                            print(f"警告: 图像格式不正确: {image.shape}")
                    else:
                        print(f"警告: 图像不是numpy数组: {type(image)}")
            return None
        except Exception as e:
            print(f"采集数据时出错: {e}")
            return None
    
    def save_frame(self, data, index):
        """保存一帧数据"""
        if data is None or 'tactile_image' not in data:
            return False
        
        try:
            image = data['tactile_image']
            timestamp = data.get('timestamp', time.time())
            
            # 保存图像
            image_path = self.save_dir / f"frame_{index:04d}.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 保存元数据
            with open(self.save_dir / f"frame_{index:04d}_meta.txt", 'w') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Frame index: {data.get('frame_index', -1)}\n")
                f.write(f"Device name: {data.get('device_name', 'unknown')}\n")
                f.write(f"Image shape: {image.shape}\n")
                f.write(f"Image dtype: {image.dtype}\n")
                f.write(f"Image min: {image.min()}\n")
                f.write(f"Image max: {image.max()}\n")
            
            self.saved_frames.append({
                'index': index,
                'timestamp': timestamp,
                'image_path': str(image_path)
            })
            
            return True
        except Exception as e:
            print(f"保存数据时出错: {e}")
            return False
    
    def run_test(self, duration=10, display=True):
        """运行测试，采集和保存数据"""
        if not self.connect():
            return False
        
        try:
            self.is_running = True
            self.frame_count = 0
            self.start_time = time.time()
            fps_update_time = self.start_time
            fps_frame_count = 0
            current_fps = 0
            
            print(f"开始测试，持续时间: {duration}秒")
            print("按 'q' 键退出测试")
            
            # 创建显示窗口
            if display:
                window_name = f"DIGIT传感器 - {self.device_name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
            
            while self.is_running:
                # 检查是否达到测试时间
                current_time = time.time()
                if duration > 0 and current_time - self.start_time >= duration:
                    print(f"\n已达到测试时间 {duration}秒，停止测试")
                    break
                
                # 采集数据
                data = self.capture_frame()
                
                if data:
                    with self.lock:
                        self.frame_count += 1
                        fps_frame_count += 1
                    
                    # 计算FPS
                    if current_time - fps_update_time >= 1.0:
                        current_fps = fps_frame_count / (current_time - fps_update_time)
                        fps_frame_count = 0
                        fps_update_time = current_time
                    
                    # 保存数据
                    self.save_frame(data, self.frame_count)
                    
                    # 显示图像
                    if display:
                        image = data['tactile_image']
                        # 添加文本信息
                        info_image = image.copy()
                        cv2.putText(info_image, f"Frame: {self.frame_count}", (10, 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(info_image, f"FPS: {current_fps:.1f}", (10, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(info_image, f"Time: {current_time - self.start_time:.1f}s", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.imshow(window_name, cv2.cvtColor(info_image, cv2.COLOR_RGB2BGR))
                        
                        # 检查键盘输入
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\n用户按下 'q' 键，停止测试")
                            break
                else:
                    # 没有数据，等待一小段时间
                    time.sleep(0.01)
            
            # 测试结束
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print("\n测试完成:")
            print(f"  总帧数: {self.frame_count}")
            print(f"  运行时间: {elapsed_time:.1f}秒")
            print(f"  平均FPS: {avg_fps:.1f}")
            print(f"  保存目录: {self.save_dir}")
            
            # 关闭窗口
            if display:
                cv2.destroyAllWindows()
            
            return True
            
        except KeyboardInterrupt:
            print("\n收到键盘中断，停止测试")
        except Exception as e:
            print(f"\n测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            self.disconnect()
        
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DIGIT传感器测试工具")
    parser.add_argument("--device", type=str, default="DIGIT", help="设备名称")
    parser.add_argument("--duration", type=int, default=10, help="测试持续时间(秒)")
    parser.add_argument("--output", type=str, default="digit_test_output", help="输出目录")
    parser.add_argument("--no-display", action="store_true", help="不显示图像窗口")
    args = parser.parse_args()
    
    # 创建测试器
    tester = DIGITTester(device_name=args.device, save_dir=args.output)
    
    # 运行测试
    tester.run_test(duration=args.duration, display=not args.no_display)

if __name__ == "__main__":
    main() 