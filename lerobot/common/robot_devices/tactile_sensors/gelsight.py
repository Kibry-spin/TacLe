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
This file contains utilities for reading tactile data from GelSight sensors.
"""

import argparse
import sys
import time
import os
import numpy as np
import cv2
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import threading

# Add the project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Try different import methods
try:
    from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
    from lerobot.common.robot_devices.utils import (
        RobotDeviceAlreadyConnectedError,
        RobotDeviceNotConnectedError,
    )
    from lerobot.common.utils.utils import capture_timestamp_utc
except ImportError:
    # Fallback for direct execution
    try:
        from configs import GelSightConfig
        # Define minimal exception classes if utils not available
        class RobotDeviceAlreadyConnectedError(Exception):
            pass
        class RobotDeviceNotConnectedError(Exception):
            pass
        # Simple timestamp function
        def capture_timestamp_utc():
            return time.time()
    except ImportError:
        # Define everything locally if needed
        from dataclasses import dataclass
        
        @dataclass
        class GelSightConfig:
            device_name: str = "GelSight Mini"
            imgh: int = 240
            imgw: int = 320
            raw_imgh: int = 2464
            raw_imgw: int = 3280
            framerate: int = 25
            config_path: str = ""
        
        class RobotDeviceAlreadyConnectedError(Exception):
            pass
        class RobotDeviceNotConnectedError(Exception):
            pass
        
        def capture_timestamp_utc():
            return time.time()


def simple_test(device_name: str = "GelSight Mini", duration_s: float = 5.0):
    """Test GelSight sensor with real-time image streaming like fast_stream_device.py."""
    print(f"Testing GelSight sensor '{device_name}' for {duration_s} seconds")
    print("Press any key to quit or wait for test to complete automatically")
    
    # Create and connect sensor
    config = GelSightConfig(device_name=device_name)
    sensor = GelSightSensor(config)
    
    try:
        sensor.connect()
        print(f"Connected to sensor: {sensor.get_sensor_info()['device_name']}")
        
        print("Starting real-time image streaming...")
        print("Image will be displayed in a window - press any key to quit")
        
        # Create window with fixed title
        window_title = f"GelSight - {device_name}"
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_title, 100, 100)  # Position window
        
        start_time = time.time()
        frame_count = 0
        fps_counter = 0
        fps_start_time = start_time
        
        while True:
            # Check time limit
            current_time = time.time()
            if current_time - start_time > duration_s:
                print(f"\nReached time limit of {duration_s} seconds")
                break
            
            # Read sensor data
            data = sensor.read()
            if data and data.get('image') is not None:
                frame_count += 1
                fps_counter += 1
                image = data['image']
                
                # Calculate FPS every second
                if current_time - fps_start_time >= 1.0:
                    actual_fps = fps_counter / (current_time - fps_start_time)
                    print(f"Frame {frame_count}: FPS: {actual_fps:.1f}, Shape: {image.shape}")
                    fps_counter = 0
                    fps_start_time = current_time
                
                # Display image in the same window
                cv2.imshow(window_title, image)
                
                # Check for keyboard input (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Any key pressed (255 means no key)
                    print(f"\nKey pressed (code: {key}), exiting...")
                    break
            else:
                print("Warning: No image data received")
                time.sleep(0.01)  # Small delay to prevent busy loop
        
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"\nTest completed:")
        print(f"  Total frames: {frame_count}")
        print(f"  Elapsed time: {elapsed_time:.1f} seconds")
        print(f"  Average FPS: {avg_fps:.1f}")
        
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping test...")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            sensor.disconnect()
        except:
            pass
        cv2.destroyAllWindows()
        print("Test cleanup completed")


def stream_test(device_name: str = "GelSight Mini"):
    """Continuous streaming test (like fast_stream_device.py) - runs until manually stopped."""
    print(f"Starting continuous GelSight streaming for '{device_name}'")
    print("Press any key in the image window to quit")
    
    # Create and connect sensor
    config = GelSightConfig(device_name=device_name)
    sensor = GelSightSensor(config)
    
    try:
        sensor.connect()
        print(f"Connected to sensor: {sensor.get_sensor_info()['device_name']}")
        print("Streaming started... (Press any key to quit)")
        
        # Create window with fixed title
        window_title = f"GelSight Stream - {device_name}"
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_title, 100, 100)  # Position window
        
        frame_count = 0
        
        while True:
            # Read sensor data
            data = sensor.read()
            if data and data.get('image') is not None:
                frame_count += 1
                image = data['image']
                
                # Display image in the same window
                cv2.imshow(window_title, image)
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Any key pressed
                    print(f"\nExiting after {frame_count} frames...")
                    break
            else:
                # Small delay if no data
                time.sleep(0.001)
        
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping stream...")
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            sensor.disconnect()
        except:
            pass
        cv2.destroyAllWindows()
        print("Streaming stopped")


class GelSightSensor:
    """
    GelSight tactile sensor implementation.
    
    Example usage:
    ```python
    config = GelSightConfig(device_name="GelSight Mini")
    sensor = GelSightSensor(config)
    sensor.connect()
    data = sensor.read()
    sensor.disconnect()
    ```
    """

    def __init__(self, config: GelSightConfig):
        self.config = config
        self.device_name = config.device_name
        self.imgh = config.imgh
        self.imgw = config.imgw
        self.raw_imgh = config.raw_imgh
        self.raw_imgw = config.raw_imgw
        self.framerate = config.framerate
        self.config_path = config.config_path

        self._device = None
        self._connected = False
        
        # --- Multithreading for non-blocking read ---
        self._latest_data = None
        self._data_lock = threading.Lock()
        self._reader_thread = None
        self._stop_event = threading.Event()
        self._has_critical_error = False  # 标记是否有关键错误
        self.logs = {}

    def connect(self):
        """Connect to the GelSight sensor with optimized settings for minimal latency."""
        if self._connected:
            raise RobotDeviceAlreadyConnectedError(f"GelSightSensor({self.device_name}) is already connected.")

        try:
            # Try to import gs_sdk
            try:
                from lerobot.common.robot_devices.tactile_sensors.gs_sdk.gs_sdk.gs_device import FastCamera
            except ImportError:
                from gs_sdk.gs_sdk.gs_device import FastCamera

            # Initialize and connect the camera
            self._device = FastCamera(
                self.device_name,
                self.imgh,
                self.imgw,
                self.raw_imgh,
                self.raw_imgw,
                self.framerate,
                verbose=False,
            )
            self._device.connect(verbose=False)
            self._connected = True
            
            # Start the background reader thread
            self._stop_event.clear()
            self._reader_thread = threading.Thread(target=self._background_reader, daemon=True)
            self._reader_thread.start()
            
            print(f"GelSightSensor({self.device_name}) connected successfully.")
            # Initial read to populate latest_data quickly
            time.sleep(1 / self.framerate * 5)  # Wait for a few frames to be sure

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to GelSightSensor({self.device_name}): {e}") from e

    def _background_reader(self):
        """Continuously reads frames from the camera in a background thread."""
        frame_index = 0
        consecutive_errors = 0
        max_consecutive_errors = 3  # 最多允许3次连续错误
        
        while not self._stop_event.is_set() and self._connected:
            try:
                # `get_image` is the blocking call from the underlying SDK
                image = self._device.get_image()
                timestamp = time.time()
                frame_index += 1
                
                if image is not None:
                    # 重置错误计数器
                    consecutive_errors = 0
                    
                    # The SDK returns a BGR image by default from ffmpeg
                    data = {
                        "tactile_image": image,
                        "timestamp": timestamp,
                        "frame_index": frame_index,
                        "device_name": self.device_name,
                    }
                    
                    with self._data_lock:
                        self._latest_data = data
                else:
                    # Reduce busy-waiting if no frame is available
                    time.sleep(1 / (self.framerate * 2))
                    
            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)
                
                # 检查是否是严重的数据错误
                is_critical_error = (
                    "cannot reshape array" in error_msg or
                    "size 0 into shape" in error_msg or
                    "Broken pipe" in error_msg or
                    "Connection reset" in error_msg
                )
                
                if is_critical_error or consecutive_errors >= max_consecutive_errors:
                    print(f"🚨 严重错误！GelSight传感器 {self.device_name} 出现关键错误:")
                    print(f"   错误信息: {error_msg}")
                    print(f"   连续错误次数: {consecutive_errors}")
                    print(f"⚠️  建议立即停止当前录制任务!")
                    
                    # 标记传感器为故障状态
                    with self._data_lock:
                        self._latest_data = {
                            "error": True,
                            "error_message": error_msg,
                            "timestamp": time.time(),
                            "frame_index": frame_index,
                            "device_name": self.device_name,
                        }
                    
                    # 设置故障标志
                    self._has_critical_error = True
                    break
                else:
                    print(f"Warning: GelSight传感器 {self.device_name} 出现错误 ({consecutive_errors}/{max_consecutive_errors}): {error_msg}")
                    time.sleep(0.1)  # 短暂等待后重试
                    
        print("GelSight background reader thread stopped.")

    def read(self) -> Optional[Dict[str, Any]]:
        """
        Reads the latest available tactile data. This is now a non-blocking call.
        It returns the 'tactile_image' field for compatibility with manipulator.py
        """
        if not self._connected:
            print("Warning: Attempted to read from a disconnected GelSight sensor.")
            return None
        
        with self._data_lock:
            # Return a copy to prevent race conditions if the caller modifies the dict
            latest_data_copy = self._latest_data.copy() if self._latest_data else None

        if latest_data_copy is None:
            # This might happen if read() is called before the first frame is captured
            return self._create_empty_data()
        
        return latest_data_copy

    def _create_empty_data(self, timestamp: float = None) -> dict:
        """Helper to create a data dictionary with null values."""
        if timestamp is None:
            timestamp = time.time()
        
        return {
            "tactile_image": np.zeros((self.imgh, self.imgw, 3), dtype=np.uint8),
            "timestamp": timestamp,
            "frame_index": -1,
            "device_name": self.device_name,
        }

    def async_read(self) -> dict:
        """Alias for read() for compatibility."""
        return self.read()

    def is_connected(self) -> bool:
        """Check if the sensor is connected."""
        return self._connected and self._reader_thread is not None and self._reader_thread.is_alive()
    
    def has_critical_error(self) -> bool:
        """检查传感器是否有关键错误"""
        return getattr(self, '_has_critical_error', False)
    
    def get_error_status(self) -> dict:
        """获取传感器错误状态"""
        if not self.has_critical_error():
            return {"has_error": False}
        
        with self._data_lock:
            if self._latest_data and self._latest_data.get("error"):
                return {
                    "has_error": True,
                    "error_message": self._latest_data.get("error_message", "Unknown error"),
                    "timestamp": self._latest_data.get("timestamp", 0),
                    "device_name": self.device_name
                }
        
        return {"has_error": True, "error_message": "Critical error detected"}

    def get_sensor_info(self) -> dict:
        """
        Get sensor information.
        """
        return {
            "device_name": self.device_name,
            "height": self.imgh,
            "width": self.imgw,
            "framerate": self.framerate,
            "is_connected": self.is_connected(),
        }

    def disconnect(self):
        """Disconnects from the sensor and cleans up resources."""
        if not self._connected:
            return

        print(f"Disconnecting GelSightSensor({self.device_name})...")
        
        # Signal the background thread to stop
        self._stop_event.set()
        
        # Wait for the thread to terminate
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0) # 1-second timeout
            if self._reader_thread.is_alive():
                print(f"Warning: GelSight reader thread for {self.device_name} did not terminate in time.")
            self._reader_thread = None

        # Release the camera device with improved cleanup
        if self._device:
            try:
                # 直接终止FFmpeg进程而不等待
                if hasattr(self._device, 'process') and self._device.process:
                    import signal
                    import subprocess
                    
                    print(f"Terminating FFmpeg process for {self.device_name}...")
                    try:
                        # 首先尝试优雅地终止进程
                        self._device.process.terminate()
                        
                        # 给进程一点时间来优雅地退出
                        try:
                            self._device.process.wait(timeout=1.0)
                            print(f"FFmpeg process terminated gracefully for {self.device_name}")
                        except subprocess.TimeoutExpired:
                            # 如果优雅终止失败，强制杀死进程
                            print(f"Force killing FFmpeg process for {self.device_name}...")
                            self._device.process.kill()
                            self._device.process.wait()  # 确保进程真正结束
                            print(f"FFmpeg process killed for {self.device_name}")
                            
                    except Exception as e:
                        print(f"Error terminating FFmpeg process: {e}")
                        # 最后尝试：查找并杀死相关的FFmpeg进程
                        try:
                            result = subprocess.run(['pgrep', '-f', f'ffmpeg.*{self._device.device}'], 
                                                  capture_output=True, text=True, timeout=2.0)
                            if result.returncode == 0 and result.stdout.strip():
                                pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
                                for pid in pids:
                                    try:
                                        subprocess.run(['kill', '-9', pid], timeout=1.0)
                                        print(f"Force killed FFmpeg process {pid}")
                                    except:
                                        pass
                        except:
                            pass
                
                # 关闭stdout（如果还打开着）
                if hasattr(self._device, 'process') and self._device.process and self._device.process.stdout:
                    try:
                        self._device.process.stdout.close()
                    except:
                        pass
                
                print(f"GelSight device {self.device_name} cleaned up successfully.")
                
            except Exception as e:
                print(f"Error during GelSight disconnect: {e}")
            finally:
                self._device = None
        
        self._connected = False
        print(f"GelSightSensor({self.device_name}) disconnected.")

    def calibrate(self):
        """
        Run sensor calibration (if applicable).
        """
        if not self._connected:
            raise RobotDeviceNotConnectedError("Cannot calibrate a disconnected sensor.")
        # TODO: Implement calibration logic if needed for GelSight
        print("Calibration is not implemented for GelSightSensor in this version.")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if getattr(self, '_connected', False):
                self.disconnect()
        except Exception as e:
            print(f"Error in GelSightSensor destructor: {e}")
            # 最后尝试强制清理
            try:
                if hasattr(self, '_device') and self._device:
                    if hasattr(self._device, 'process') and self._device.process:
                        try:
                            self._device.process.kill()
                            self._device.process.wait()
                        except:
                            pass
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GelSight tactile sensor test and streaming utility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gelsight.py                                    # Test with default settings (5 seconds)
  python gelsight.py --device "GelSight Mini" --duration 10  # Test specific device for 10 seconds
  python gelsight.py --stream                           # Continuous streaming mode
  python gelsight.py --stream --device "GelSight Mini"  # Stream specific device
        """
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="GelSight Mini",
        help="Device name for the GelSight sensor (default: 'GelSight Mini')."
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
        help="Enable continuous streaming mode (like fast_stream_device.py). Runs until manually stopped."
    )
    
    args = parser.parse_args()
    
    if args.stream:
        stream_test(device_name=args.device)
    else:
        simple_test(
            device_name=args.device,
            duration_s=args.duration
        ) 