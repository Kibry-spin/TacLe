# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
提供DIGIT触觉传感器接口，基于digit-interface库实现图像获取和处理。
"""

import sys
import time
import os
import numpy as np
import cv2
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
import threading
import pprint

# Add the project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import dependencies
try:
    from lerobot.common.robot_devices.tactile_sensors.configs import DIGITConfig
    from lerobot.common.robot_devices.utils import (
        RobotDeviceAlreadyConnectedError,
        RobotDeviceNotConnectedError,
    )
except ImportError:
    # Fallback not available – inform user to install proper dependencies
    raise ImportError(
        "Required modules not found. Please ensure lerobot package is installed correctly "
        "and that you are executing within the project environment."
    )

# Import digit-interface library
try:
    # Try direct import from digit-interface directory
    sys.path.append(str(Path(__file__).parent / "digit-interface"))
    from digit_interface.digit import Digit
    from digit_interface.digit_handler import DigitHandler
except ImportError:
    print("Warning: digit-interface library not found. Please check the installation.")
    Digit = None
    DigitHandler = None


def simple_test(device_name: str = "DIGIT", duration_s: float = 5.0):
    """测试DIGIT传感器，实时显示图像流"""
    print(f"Testing DIGIT sensor '{device_name}' for {duration_s} seconds")
    print("Press any key to quit or wait for test to complete automatically")
    
    # 检查 DigitHandler 是否可用
    if DigitHandler is None:
        print("Error: digit-interface library not found or not properly initialized")
        return
    
    # 列出所有可用的 DIGIT 设备
    try:
        digits = DigitHandler.list_digits()
        print("Connected DIGIT's to Host:")
        pprint.pprint(digits)
    except Exception as e:
        print(f"Error listing DIGIT devices: {e}")
        return
    
    # 创建并连接传感器
    if device_name == "DIGIT" and digits:
        # 使用第一个可用设备
        serial_number = digits[0]["serial"]
        print(f"Using first available DIGIT device: {serial_number}")
        device_name = serial_number
    
    # 检查 Digit 类是否可用
    if Digit is None:
        print("Error: Digit class not available")
        return
    
    try:
        # 创建 Digit 对象并直接连接，不使用 DIGITSensor 类
        digit = Digit(serial=device_name, name=f"DIGIT_{device_name}")
        digit.connect()
    except Exception as e:
        print(f"Error connecting to DIGIT device: {e}")
        return
    
    # 打印设备信息
    print(digit.info())
    
    # 设置 LED 照明：先关闭再打开（预热流程）
    try:
        digit.set_intensity(0)  # 最小亮度
        time.sleep(0.2)
        digit.set_intensity(15)  # 最大亮度
    except Exception as e:
        print(f"Warning: Could not set LED intensity: {e}")
    
    # 设置分辨率为 QVGA (320x240)
    try:
        # 直接使用已知的 QVGA 参数
        qvga_res = {"resolution": {"width": 320, "height": 240}}
        digit.set_resolution(qvga_res)
    except Exception as e:
        print(f"Warning: Could not set resolution: {e}")
    
    # 设置帧率为 60fps
    try:
        digit.set_fps(60)
    except Exception as e:
        print(f"Warning: Could not set framerate: {e}")
    
    print("Starting real-time image streaming...")
    print("Image will be displayed in a window - press any key to quit")
    
    # 创建窗口
    window_title = f"DIGIT - {device_name}"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_title, 100, 100)  # 窗口位置
    
    start_time = time.time()
    frame_count = 0
    fps_counter = 0
    fps_start_time = start_time
    
    while True:
        # 检查时间限制
        current_time = time.time()
        if current_time - start_time > duration_s:
            print(f"\nReached time limit of {duration_s} seconds")
            break
        
        # 直接从 Digit 对象获取图像
        frame = digit.get_frame()
        frame_count += 1
        fps_counter += 1
        
        # 计算 FPS
        if current_time - fps_start_time >= 1.0:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Frame {frame_count}: FPS: {actual_fps:.1f}, Shape: {frame.shape}")
            fps_counter = 0
            fps_start_time = current_time
        
        # 显示图像
        cv2.imshow(window_title, frame)
        
        # 检查键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # 按下任意键
            print(f"\nKey pressed (code: {key}), exiting...")
            break
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"\nTest completed:")
    print(f"  Total frames: {frame_count}")
    print(f"  Elapsed time: {elapsed_time:.1f} seconds")
    print(f"  Average FPS: {avg_fps:.1f}")
    
    # 断开连接
    digit.disconnect()
    cv2.destroyAllWindows()
    print("Test cleanup completed")


def stream_test(device_name: str = "DIGIT"):
    """持续流模式 - 运行直到手动停止"""
    print(f"Starting continuous DIGIT streaming for '{device_name}'")
    print("Press any key in the image window to quit")
    
    # 检查 DigitHandler 是否可用
    if DigitHandler is None:
        print("Error: digit-interface library not found or not properly initialized")
        return
    
    # 列出所有可用的 DIGIT 设备
    try:
        digits = DigitHandler.list_digits()
        if not digits:
            print("No DIGIT devices found!")
            return
    except Exception as e:
        print(f"Error listing DIGIT devices: {e}")
        return
    
    if device_name == "DIGIT":
        # 使用第一个可用设备
        serial_number = digits[0]["serial"]
        print(f"Using first available DIGIT device: {serial_number}")
        device_name = serial_number
    
    # 检查 Digit 类是否可用
    if Digit is None:
        print("Error: Digit class not available")
        return
    
    try:
        # 创建 Digit 对象并直接连接
        digit = Digit(serial=device_name, name=f"DIGIT_{device_name}")
        digit.connect()
    except Exception as e:
        print(f"Error connecting to DIGIT device: {e}")
        return
    
    # 打印设备信息
    print(digit.info())
    
    # 设置 LED 照明：先关闭再打开（预热流程）
    try:
        digit.set_intensity(0)  # 最小亮度
        time.sleep(0.2)
        digit.set_intensity(15)  # 最大亮度
    except Exception as e:
        print(f"Warning: Could not set LED intensity: {e}")
    
    # 设置分辨率为 QVGA (320x240)
    try:
        # 直接使用已知的 QVGA 参数
        qvga_res = {"resolution": {"width": 320, "height": 240}}
        digit.set_resolution(qvga_res)
    except Exception as e:
        print(f"Warning: Could not set resolution: {e}")
    
    # 设置帧率为 60fps
    try:
        digit.set_fps(60)
    except Exception as e:
        print(f"Warning: Could not set framerate: {e}")
    
    print("Streaming started... (Press any key to quit)")
    
    # 创建窗口
    window_title = f"DIGIT Stream - {device_name}"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_title, 100, 100)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 直接从 Digit 对象获取图像
            frame = digit.get_frame()
            frame_count += 1
            
            # 每 100 帧显示一次 FPS
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Frame {frame_count}, Average FPS: {fps:.1f}")
            
            # 显示图像
            cv2.imshow(window_title, frame)
            
            # 检查键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # 按下任意键
                print(f"\nExiting after {frame_count} frames...")
                break
    
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping stream...")
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback; traceback.print_exc()
    finally:
        # 断开连接
        digit.disconnect()
        cv2.destroyAllWindows()
        print("Streaming stopped")


class DIGITSensor:
    """
    DIGIT触觉传感器实现，使用digit-interface库。
    
    Example usage:
    ```python
    config = DIGITConfig(device_name="D21186")  # Use actual serial number
    sensor = DIGITSensor(config)
    sensor.connect()
    data = sensor.read()
    sensor.disconnect()
    ```
    """

    def __init__(self, config: DIGITConfig):
        self.config = config
        self.device_name = config.device_name
        self.imgh = config.imgh
        self.imgw = config.imgw
        self.framerate = config.framerate

        self._device = None
        self._connected = False
        
        # --- Multithreading for non-blocking read ---
        self._latest_data = None
        self._data_lock = threading.Lock()
        self._reader_thread = None
        self._stop_event = threading.Event()
        self._has_critical_error = False

    def connect(self):
        """Connect to the DIGIT sensor using digit-interface library."""
        if self._connected:
            raise RobotDeviceAlreadyConnectedError(f"DIGITSensor({self.device_name}) is already connected.")

        if Digit is None or DigitHandler is None:
            raise ImportError("digit-interface library not found. Please install it first.")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"DIGIT连接尝试 {attempt + 1}/{max_retries}...")
                
                # 列出所有可用的 DIGIT 设备
                available_digits = DigitHandler.list_digits()
                if not available_digits:
                    raise ConnectionError("No DIGIT devices found")
                
                # 确定要使用的设备序列号
                if self.device_name and len(self.device_name) > 5:
                    # 假设是序列号
                    serial_number = self.device_name
                else:
                    # 使用第一个可用设备
                    serial_number = available_digits[0]["serial"]
                    print(f"Using first available DIGIT device: {serial_number}")
                
                # 创建 Digit 设备实例
                self._device = Digit(serial=serial_number, name=f"DIGIT_{serial_number}")
                self._device.connect()
                
                # 设置 LED 照明：先关闭再打开（预热流程）
                self._device.set_intensity(Digit.LIGHTING_MIN)
                time.sleep(0.2)
                self._device.set_intensity(Digit.LIGHTING_MAX)
                
                # The digit-interface library sets stream defaults (QVGA, 60fps) on connect.
                # We rely on these defaults to align with the demo script's behavior.
                
                # 测试连接 - 尝试获取多帧来确保稳定
                print("测试DIGIT设备连接...")
                for test_frame in range(3):
                    test_image = self._device.get_frame()
                    if test_image is None:
                        raise ConnectionError(f"Failed to get test frame {test_frame + 1}")
                    time.sleep(0.1)
                
                print(f"DIGIT sensor connected successfully. Device info: {self._device.info()}")
                
                # 在启动线程前设置_connected为True
                self._connected = True
                self._has_critical_error = False
                
                # 启动后台读取线程
                self._stop_event.clear()
                self._reader_thread = threading.Thread(target=self._background_reader, daemon=True)
                self._reader_thread.start()
                
                # 等待后台线程启动并获取第一帧数据
                print("等待后台线程启动...")
                for wait_attempt in range(20):  # 最多等待10秒
                    time.sleep(0.5)
                    if self._latest_data is not None and self._latest_data.get('frame_index', -1) > 0:
                        print(f"后台线程启动成功，获得第一帧: frame_index={self._latest_data['frame_index']}")
                        break
                    if not self._reader_thread.is_alive():
                        raise ConnectionError("Background reader thread died during startup")
                else:
                    if self._latest_data is None or self._latest_data.get('frame_index', -1) <= 0:
                        raise ConnectionError("Background reader thread failed to start or get data")
                
                print(f"Connected to DIGIT sensor: {self.device_name}")
                return  # 成功连接，退出重试循环
                
            except Exception as e:
                print(f"DIGIT连接尝试 {attempt + 1} 失败: {e}")
                
                # 清理失败的连接
                self._connected = False
                try:
                    if self._device:
                        self._device.disconnect()
                        self._device = None
                except:
                    pass
                
                if self._reader_thread and self._reader_thread.is_alive():
                    self._stop_event.set()
                    self._reader_thread.join(timeout=1.0)
                    self._reader_thread = None
                
                if attempt < max_retries - 1:
                    print(f"等待 2 秒后重试...")
                    time.sleep(2)
                else:
                    self._has_critical_error = True
                    error_msg = f"Failed to connect to DIGIT sensor after {max_retries} attempts: {e}"
                    print(error_msg)
                    raise ConnectionError(error_msg) from e

    def _background_reader(self):
        """Continuously reads frames from the camera in a background thread."""
        frame_index = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self._stop_event.is_set() and self._connected:
            try:
                # 直接从 Digit 对象获取图像
                if self._device is None:
                    time.sleep(0.1)
                    continue
                
                # 获取原始帧（BGR 格式）
                image = self._device.get_frame()
                timestamp = time.time()
                frame_index += 1
                
                if image is not None:
                    # 重置错误计数器
                    consecutive_errors = 0
                    
                    # 确保图像格式正确
                    if isinstance(image, np.ndarray):
                        # 转换 BGR 到 RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # 确保数据类型正确
                        if image.dtype != np.uint8:
                            image = image.astype(np.uint8)
                        
                        # 调整尺寸和方向
                        if image.shape[:2] == (self.imgw, self.imgh):
                            # 图像是旋转的 (e.g., shape is 320x240 but should be 240x320), 顺时针旋转90度
                            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        elif image.shape[:2] != (self.imgh, self.imgw):
                            # 尺寸不匹配，进行缩放
                            image = cv2.resize(image, (self.imgw, self.imgh))
                    else:
                        print(f"Warning: DIGIT image is not numpy array: {type(image)}")
                        image = np.zeros((self.imgh, self.imgw, 3), dtype=np.uint8)
                    
                    # 创建数据字典
                    data = {
                        "tactile_image": image,
                        "timestamp": timestamp,
                        "frame_index": frame_index,
                        "device_name": self.device_name,
                    }
                    
                    with self._data_lock:
                        self._latest_data = data
                else:
                    time.sleep(1 / (self.framerate * 2))
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Warning: DIGIT sensor {self.device_name} failed to get image {consecutive_errors} times in a row")
                        consecutive_errors = 0
            
            except Exception as e:
                print(f"Error in DIGIT background reader: {e}")
                consecutive_errors += 1
                time.sleep(0.1)  # 延迟以防止 CPU 过载
                
                if consecutive_errors >= max_consecutive_errors:
                    self._has_critical_error = True
                    print(f"Critical error: DIGIT sensor {self.device_name} background reader failed too many times")
                    break

    def read(self) -> Optional[Dict[str, Any]]:
        """
        读取最新的触觉数据（非阻塞调用）
        """
        if not self._connected:
            print("Warning: Attempted to read from a disconnected DIGIT sensor.")
            return None
        
        with self._data_lock:
            # 返回副本以防止调用者修改字典时的竞争条件
            latest_data_copy = self._latest_data.copy() if self._latest_data else None

        if latest_data_copy is None:
            # 这可能发生在第一帧捕获之前调用 read() 时
            return self._create_empty_data()
        
        return latest_data_copy

    def _create_empty_data(self, timestamp: Optional[float] = None) -> dict:
        """创建空数据字典"""
        if timestamp is None:
            timestamp = time.time()
        
        return {
            "tactile_image": np.zeros((self.imgh, self.imgw, 3), dtype=np.uint8),
            "timestamp": timestamp,
            "frame_index": -1,
            "device_name": self.device_name,
        }

    def async_read(self) -> Optional[dict]:
        """read()方法的别名，保持兼容性"""
        return self.read()

    def is_connected(self) -> bool:
        """检查传感器是否已连接"""
        return self._connected and self._reader_thread is not None and self._reader_thread.is_alive()
    
    def has_critical_error(self) -> bool:
        """检查传感器是否有严重错误"""
        return self._has_critical_error
    
    def get_sensor_info(self) -> dict:
        """获取传感器信息"""
        info = {
            "device_name": self.device_name,
            "resolution": f"{self.imgw}x{self.imgh}",
            "framerate": self.framerate,
            "is_connected": self.is_connected(),
        }
        
        # 添加设备特定信息（如果可用）
        if self._device and self._connected:
            try:
                device_info = self._device.info()
                info["device_info"] = device_info
            except:
                pass
        
        return info

    # def disconnect(self):
    #     """断开传感器连接"""
    #     if not self._connected:
    #         return

    #     # 停止后台线程
    #     self._stop_event.set()
    #     if self._reader_thread and self._reader_thread.is_alive():
    #         self._reader_thread.join(timeout=1.0)
        
    #     # 释放相机
    #     if self._device:
    #         try:
    #             self._device.disconnect()
    #         except Exception as e:
    #             print(f"Error releasing DIGIT device: {e}")
        
    #     self._connected = False
    #     self._device = None
    #     print(f"Disconnected from DIGIT sensor: {self.device_name}")
    def disconnect(self):
        """断开传感器连接"""
        if not self._connected:
            return

        # 1. Signal the background thread to stop and mark as disconnected.
        self._stop_event.set()
        self._connected = False

        # 2. Disconnect the underlying device. This should help unblock the background
        #    thread if it's stuck in a blocking call like `get_frame()`.
        if self._device:
            try:
                self._device.disconnect()
            except Exception as e:
                print(f"Error during DIGIT device disconnect: {e}")

        # 3. Wait for the background thread to terminate.
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)  # Use a reasonable timeout
            if self._reader_thread.is_alive():
                print(f"Warning: DIGIT background reader thread for {self.device_name} did not terminate gracefully.")

        self._device = None
        self._reader_thread = None
        print(f"Disconnected from DIGIT sensor: {self.device_name}")
    def calibrate(self):
        """校准传感器（占位符 - DIGIT不需要校准）"""
        print(f"Note: DIGIT sensor {self.device_name} does not require calibration")
        return True

    def __del__(self):
        """析构函数，确保资源释放"""
        try:
            if self._connected:
                self.disconnect()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DIGIT tactile sensor test and streaming utility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python digit.py                                  # Test with default settings (5 seconds)
  python digit.py --device D21186 --duration 10    # Test specific device for 10 seconds
  python digit.py --stream                         # Continuous streaming mode
  python digit.py --stream --device D21186         # Stream specific device
        """
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="DIGIT",
        help="Device serial/name for the DIGIT sensor (default: 'DIGIT')."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Test duration in seconds (default: 5.0). Ignored in stream mode."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable continuous streaming mode. Runs until manually stopped."
    )
    
    args = parser.parse_args()
    
    if args.stream:
        stream_test(device_name=args.device)
    else:
        simple_test(device_name=args.device, duration_s=args.duration)
