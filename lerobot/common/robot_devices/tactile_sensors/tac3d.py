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
This file contains utilities for reading tactile data from Tac3D sensors.
"""

import argparse
import sys
import time
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

# Add the project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Try different import methods
try:
    from lerobot.common.robot_devices.tactile_sensors.configs import Tac3DConfig
    from lerobot.common.robot_devices.utils import (
        RobotDeviceAlreadyConnectedError,
        RobotDeviceNotConnectedError,
    )
    from lerobot.common.utils.utils import capture_timestamp_utc
except ImportError:
    # Fallback for direct execution
    try:
        from configs import Tac3DConfig
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
        class Tac3DConfig:
            port: int = 9988
            auto_calibrate: bool = True
        
        class RobotDeviceAlreadyConnectedError(Exception):
            pass
        class RobotDeviceNotConnectedError(Exception):
            pass
        
        def capture_timestamp_utc():
            return time.time()


def simple_test(port: int = 9988, duration_s: float = 5.0, calibrate: bool = True):
    """Simple test function for Tac3D sensor."""
    print(f"Testing Tac3D sensor on port {port} for {duration_s} seconds")
    
    # Create and connect sensor
    config = Tac3DConfig(port=port, auto_calibrate=calibrate)
    sensor = Tac3DSensor(config)
    
    try:
        sensor.connect()
        print(f"Connected to sensor: {sensor.get_sensor_info()['sensor_sn']}")
        
        print("Reading sensor data...")
        print("Frame | Time   | Force X | Force Y | Force Z | Magnitude")
        print("-" * 55)
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_s:
            data = sensor.read()
            if data:
                frame_count += 1
                timestamp = data.get('timestamp', 0)
                
                # Display force data
                force = data.get('resultant_force')
                if force is not None and force.size >= 3:
                    fx, fy, fz = force[0, 0], force[0, 1], force[0, 2]
                    magnitude = np.sqrt(fx*fx + fy*fy + fz*fz)
                    print(f"{frame_count:5d} | {timestamp:6.2f} | {fx:7.3f} | {fy:7.3f} | {fz:7.3f} | {magnitude:9.3f}")
                else:
                    print(f"{frame_count:5d} | {timestamp:6.2f} | No force data available")
            
            time.sleep(0.1)
        
        print(f"\nTest completed. Read {frame_count} frames in {duration_s} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sensor.disconnect()
        print("Sensor disconnected")


class Tac3DSensor:
    """
    Tac3D tactile sensor implementation.
    
    Example usage:
    ```python
    config = Tac3DConfig(port=9988, auto_calibrate=True)
    sensor = Tac3DSensor(config)
    sensor.connect()
    data = sensor.read()
    sensor.disconnect()
    ```
    """

    def __init__(self, config: Tac3DConfig):
        self.config = config
        self.port = config.port
        self.auto_calibrate = config.auto_calibrate

        self._sensor = None
        self._connected = False
        self._latest_frame = None
        self.logs = {}

    def connect(self):
        """Connect to the Tac3D sensor."""
        if self._connected:
            raise RobotDeviceAlreadyConnectedError(f"Tac3DSensor(port={self.port}) is already connected.")

        try:
            from lerobot.common.robot_devices.tactile_sensors.TAC3D.PyTac3D import Sensor
        except ImportError:
            # Try relative import for direct execution
            try:
                from TAC3D.PyTac3D import Sensor
            except ImportError:
                # Try current directory
                import os
                current_dir = os.path.dirname(__file__)
                sys.path.append(os.path.join(current_dir, 'TAC3D'))
                from PyTac3D import Sensor
        
        # Create sensor with callback to store latest frame
        self._sensor = Sensor(
            recvCallback=self._frame_callback,
            port=self.port,
            maxQSize=5
        )
        
        # Wait for sensor to be ready
        print(f"Waiting for Tac3D sensor on port {self.port}...")
        self._sensor.waitForFrame()
        self._connected = True
        
        # Auto-calibrate if enabled
        if self.auto_calibrate:
            self.calibrate()

    def _frame_callback(self, frame: Dict[str, Any], param: Any = None):
        """Callback to store the latest frame."""
        self._latest_frame = frame

    def read(self) -> dict:
        """Read tactile data from the sensor."""
        if not self._connected:
            raise RobotDeviceNotConnectedError(f"Tac3DSensor(port={self.port}) is not connected.")

        start_time = time.perf_counter()

        # Get frame data - prefer latest from callback, fallback to manual read
        frame = self._latest_frame or self._sensor.getFrame()
        
        if frame is None:
            return {}

        # Log timing information
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        # Return standardized data format
        return {
            'timestamp': frame.get('recvTimestamp', time.time()),
            'sensor_sn': frame.get('SN', 'unknown'),
            'frame_index': frame.get('index', 0),
            'positions_3d': frame.get('3D_Positions'),
            'displacements_3d': frame.get('3D_Displacements'), 
            'forces_3d': frame.get('3D_Forces'),
            'resultant_force': frame.get('3D_ResultantForce'),
            'resultant_moment': frame.get('3D_ResultantMoment'),
            'raw_frame': frame
        }

    def async_read(self) -> dict:
        """Asynchronously read tactile data from the sensor (same as read for Tac3D)."""
        return self.read()

    def is_connected(self) -> bool:
        """Check if sensor is connected."""
        return self._connected

    def calibrate(self):
        """Calibrate the sensor (zero-point calibration)."""
        if not self._connected:
            raise RobotDeviceNotConnectedError(f"Tac3DSensor(port={self.port}) is not connected.")

        # Get sensor SN from latest frame
        frame = self._latest_frame or self._sensor.getFrame()
        if frame:
            sn = frame.get('SN', 'unknown')
            print(f"Calibrating sensor {sn}...")
            self._sensor.calibrate(sn)
            
            # Wait for calibration to complete (minimum 6 seconds)
            print("Waiting for calibration to complete...")
            time.sleep(6.0)
            print("Calibration completed")
        else:
            print("Warning: No sensor data available for calibration")

    def get_sensor_info(self) -> dict:
        """Get sensor information."""
        info = {
            'port': self.port,
            'is_connected': self._connected,
        }
        
        # Add dynamic information if connected
        if self._connected and self._latest_frame:
            info.update({
                'sensor_sn': self._latest_frame.get('SN', 'unknown'),
                'last_frame_index': self._latest_frame.get('index', 0),
                'last_timestamp': self._latest_frame.get('recvTimestamp', 0.0),
            })
        
        # Add firmware version if available
        try:
            from lerobot.common.robot_devices.tactile_sensors.TAC3D.PyTac3D import PYTAC3D_VERSION
            info['firmware_version'] = PYTAC3D_VERSION
        except ImportError:
            try:
                from TAC3D.PyTac3D import PYTAC3D_VERSION
                info['firmware_version'] = PYTAC3D_VERSION
            except ImportError:
                try:
                    from PyTac3D import PYTAC3D_VERSION
                    info['firmware_version'] = PYTAC3D_VERSION
                except ImportError:
                    info['firmware_version'] = 'unknown'
        
        return info

    def disconnect(self):
        """Disconnect from the sensor."""
        if not self._connected:
            raise RobotDeviceNotConnectedError(f"Tac3DSensor(port={self.port}) is not connected.")

        if self._sensor and hasattr(self._sensor, '_UDP') and hasattr(self._sensor._UDP, 'close'):
            self._sensor._UDP.close()
        
        self._sensor = None
        self._connected = False

    def __del__(self):
        if getattr(self, "_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple Tac3D tactile sensor test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tac3d.py                           # Test with default settings
  python tac3d.py --port 9989 --duration 10 # Test specific port for 10 seconds
  python tac3d.py --no-calibrate            # Test without calibration
        """
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=9988,
        help="UDP port for the Tac3D sensor (default: 9988)."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Test duration in seconds (default: 5.0)."
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable auto-calibration on connection."
    )
    
    args = parser.parse_args()
    
    simple_test(
        port=args.port,
        duration_s=args.duration,
        calibrate=not args.no_calibrate
    ) 