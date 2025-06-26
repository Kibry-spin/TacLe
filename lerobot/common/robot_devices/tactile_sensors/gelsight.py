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
        while not self._stop_event.is_set() and self._connected:
            try:
                # `get_image` is the blocking call from the underlying SDK
                image = self._device.get_image()
                timestamp = time.time()
                frame_index += 1
                
                if image is not None:
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
                # If the device is disconnected externally, the thread should stop
                print(f"Error in GelSight background reader thread: {e}. Stopping thread.")
                break
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

        # Release the camera device
        if self._device:
            try:
                # Use a timeout mechanism for release to avoid hangs
                def release_with_timeout():
                    try:
                        self._device.release()
                        print(f"GelSight device {self.device_name} released.")
                    except Exception as e:
                        print(f"Warning: Exception during GelSight device release: {e}")

                release_thread = threading.Thread(target=release_with_timeout, daemon=True)
                release_thread.start()
                release_thread.join(timeout=2.0) # 2-second timeout for release

                if release_thread.is_alive():
                    print(f"Error: GelSight device {self.device_name} release timed out. The resource might not be freed correctly.")
                
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
            if self._connected:
                self.disconnect()
        except Exception as e:
            print(f"Error in GelSightSensor destructor: {e}")


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