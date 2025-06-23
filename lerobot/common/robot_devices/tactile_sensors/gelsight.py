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
    """Simple test function for GelSight sensor."""
    print(f"Testing GelSight sensor '{device_name}' for {duration_s} seconds")
    
    # Create and connect sensor
    config = GelSightConfig(device_name=device_name)
    sensor = GelSightSensor(config)
    
    try:
        sensor.connect()
        print(f"Connected to sensor: {sensor.get_sensor_info()['device_name']}")
        
        print("Reading sensor data...")
        print("Press 'q' to quit early or wait for test to complete")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_s:
            data = sensor.read()
            if data and data.get('image') is not None:
                frame_count += 1
                timestamp = data.get('timestamp', 0)
                image = data['image']
                
                # Display image
                cv2.imshow(f"GelSight - {device_name}", image)
                
                # Print frame info
                if frame_count % 10 == 0:  # Print every 10th frame
                    print(f"Frame {frame_count}: {timestamp:.2f}s, Shape: {image.shape}")
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                time.sleep(0.01)
        
        print(f"\nTest completed. Read {frame_count} frames in {time.time() - start_time:.1f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sensor.disconnect()
        cv2.destroyAllWindows()
        print("Sensor disconnected")


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
        self._frame_count = 0
        self.logs = {}

    def connect(self):
        """Connect to the GelSight sensor."""
        if self._connected:
            raise RobotDeviceAlreadyConnectedError(f"GelSightSensor({self.device_name}) is already connected.")

        try:
            # Try to import gs_sdk
            try:
                from lerobot.common.robot_devices.tactile_sensors.gs_sdk.gs_device import FastCamera
            except ImportError:
                # Try relative import
                sys.path.append(os.path.join(os.path.dirname(__file__), 'gs_sdk'))
                from gs_sdk.gs_device import FastCamera
            
            # Load config from file if provided
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    file_config = yaml.safe_load(f)
                    # Override with file config
                    self.device_name = file_config.get("device_name", self.device_name)
                    self.imgh = file_config.get("imgh", self.imgh)
                    self.imgw = file_config.get("imgw", self.imgw)
                    self.raw_imgh = file_config.get("raw_imgh", self.raw_imgh)
                    self.raw_imgw = file_config.get("raw_imgw", self.raw_imgw)
                    self.framerate = file_config.get("framerate", self.framerate)

            # Create and connect the device
            self._device = FastCamera(
                self.device_name, 
                self.imgh, 
                self.imgw, 
                self.raw_imgh, 
                self.raw_imgw, 
                self.framerate
            )
            self._device.connect()
            self._connected = True
            
            print(f"GelSight sensor '{self.device_name}' connected successfully")
            
        except ImportError as e:
            raise ImportError(f"Failed to import gs_sdk. Make sure it's installed: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to GelSight sensor: {e}")

    def read(self) -> dict:
        """Read tactile image data from the sensor."""
        if not self._connected:
            raise RobotDeviceNotConnectedError(f"GelSightSensor({self.device_name}) is not connected.")

        start_time = time.perf_counter()

        try:
            # Get image from device
            image = self._device.get_image()
            
            # Debug: Print image info for troubleshooting
            if image is None:
                # Try alternative method or return empty data
                print(f"Warning: GelSight sensor returned None image (attempt {self._frame_count + 1})")
                return self._create_empty_data()
            
            # Validate image data
            if not isinstance(image, np.ndarray):
                print(f"Warning: GelSight sensor returned non-numpy data: {type(image)}")
                return self._create_empty_data()
            
            if image.size == 0:
                print(f"Warning: GelSight sensor returned empty image array")
                return self._create_empty_data()
            
            # Validate image shape
            expected_shapes = [
                (self.imgh, self.imgw, 3),      # Processed image
                (self.raw_imgh, self.raw_imgw, 3),  # Raw image
                (self.imgh, self.imgw),         # Grayscale processed
                (self.raw_imgh, self.raw_imgw), # Grayscale raw
            ]
            
            if image.shape not in expected_shapes:
                print(f"Warning: Unexpected image shape: {image.shape}, expected one of {expected_shapes}")
                # Try to reshape if possible
                if image.size == self.imgh * self.imgw * 3:
                    print(f"Attempting to reshape to ({self.imgh}, {self.imgw}, 3)")
                    try:
                        image = image.reshape(self.imgh, self.imgw, 3)
                    except Exception as reshape_error:
                        print(f"Reshape failed: {reshape_error}")
                        return self._create_empty_data()
                else:
                    print(f"Cannot reshape: image size {image.size} doesn't match expected size {self.imgh * self.imgw * 3}")
                    return self._create_empty_data()

            # Ensure image is in correct format (HWC, uint8)
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image, image, image], axis=-1)  # Convert to RGB
            
            if image.dtype != np.uint8:
                if image.dtype in [np.float32, np.float64]:
                    # Assume normalized values [0,1] or [0,255]
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = np.clip(image, 0, 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            self._frame_count += 1
            current_time = time.time()

            # Log timing information
            self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
            self.logs["timestamp_utc"] = capture_timestamp_utc()

            # Return standardized data format compatible with LeRobot
            return {
                # LeRobot standard fields for tactile sensors
                'SN': self.device_name,  # Sensor serial number (use device name)
                'index': self._frame_count,  # Frame index
                'sendTimestamp': current_time,  # Send timestamp
                'recvTimestamp': current_time,  # Receive timestamp (same for local device)
                
                # GelSight specific data
                'tactile_image': image,  # Main tactile image data
                'image_shape': image.shape,
                
                # Original format for backward compatibility
                'timestamp': current_time,
                'device_name': self.device_name,
                'frame_index': self._frame_count,
                'image': image,
                'sensor_config': {
                    'imgh': self.imgh,
                    'imgw': self.imgw,
                    'framerate': self.framerate,
                    'raw_imgh': self.raw_imgh,
                    'raw_imgw': self.raw_imgw,
                }
            }
            
        except KeyboardInterrupt:
            # Handle KeyboardInterrupt specifically to ensure proper cleanup
            print(f"KeyboardInterrupt received while reading from GelSight sensor {self.device_name}")
            # Force disconnect to release resources
            try:
                self.disconnect()
            except:
                pass
            # Re-raise the KeyboardInterrupt to allow proper program termination
            raise
        except Exception as e:
            print(f"Error reading from GelSight sensor: {e}")
            return self._create_empty_data()

    def _create_empty_data(self) -> dict:
        """Create empty data structure when sensor read fails."""
        current_time = time.time()
        empty_image = np.zeros((self.imgh, self.imgw, 3), dtype=np.uint8)
        
        return {
            # LeRobot standard fields
            'SN': self.device_name,
            'index': self._frame_count,
            'sendTimestamp': current_time,
            'recvTimestamp': current_time,
            
            # GelSight specific data
            'tactile_image': empty_image,
            'image_shape': empty_image.shape,
            
            # Original format for backward compatibility
            'timestamp': current_time,
            'device_name': self.device_name,
            'frame_index': self._frame_count,
            'image': empty_image,
            'sensor_config': {
                'imgh': self.imgh,
                'imgw': self.imgw,
                'framerate': self.framerate,
                'raw_imgh': self.raw_imgh,
                'raw_imgw': self.raw_imgw,
            }
        }

    def async_read(self) -> dict:
        """Asynchronously read tactile data from the sensor (same as read for GelSight)."""
        return self.read()

    def is_connected(self) -> bool:
        """Check if sensor is connected."""
        return self._connected

    def get_sensor_info(self) -> dict:
        """Get sensor information."""
        info = {
            'device_name': self.device_name,
            'is_connected': self._connected,
            'imgh': self.imgh,
            'imgw': self.imgw,
            'raw_imgh': self.raw_imgh,
            'raw_imgw': self.raw_imgw,
            'framerate': self.framerate,
        }
        
        # Add dynamic information if connected
        if self._connected:
            info.update({
                'frame_count': self._frame_count,
            })
        
        return info

    def disconnect(self):
        """Disconnect from the sensor."""
        if not self._connected:
            print(f"GelSightSensor({self.device_name}) is already disconnected.")
            return

        try:
            if self._device and hasattr(self._device, 'release'):
                print(f"Releasing GelSight device {self.device_name}...")
                self._device.release()
                print(f"GelSight device {self.device_name} released successfully.")
        except Exception as e:
            print(f"Warning: Error during GelSight sensor disconnect: {e}")
        finally:
            self._device = None
            self._connected = False
            print(f"GelSightSensor({self.device_name}) disconnected.")

    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            if getattr(self, "_connected", False):
                self.disconnect()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple GelSight tactile sensor test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gelsight.py                                    # Test with default settings
  python gelsight.py --device "GelSight Mini" --duration 10  # Test specific device for 10 seconds
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
        help="Test duration in seconds (default: 5.0)."
    )
    
    args = parser.parse_args()
    
    simple_test(
        device_name=args.device,
        duration_s=args.duration
    ) 