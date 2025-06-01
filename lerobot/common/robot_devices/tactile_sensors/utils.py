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

from typing import Protocol, Dict, Any, Optional

import numpy as np

from lerobot.common.robot_devices.tactile_sensors.configs import (
    TactileSensorConfig,
    Tac3DConfig,
)


# Defines a tactile sensor interface
class TactileSensor(Protocol):
    """Protocol defining the interface for tactile sensors."""
    
    def connect(self) -> None:
        """Connect to the tactile sensor."""
        ...
    
    def read(self) -> Dict[str, Any]:
        """
        Read tactile data from the sensor.
        
        Returns:
            Dict containing sensor data with standardized keys:
            - 'timestamp': float, time when data was captured
            - 'sensor_id': str, unique identifier for the sensor
            - 'positions_3d': np.ndarray, 3D positions of tactile points (if available)
            - 'displacements_3d': np.ndarray, 3D displacements (if available)  
            - 'forces_3d': np.ndarray, 3D forces at tactile points (if available)
            - 'resultant_force': np.ndarray, resultant force vector (if available)
            - 'resultant_moment': np.ndarray, resultant moment vector (if available)
            - 'pressure': np.ndarray, pressure values (if available)
            - 'temperature': float, sensor temperature (if available)
            - 'contact_area': float, contact area (if available)
            - 'raw_data': Any, original sensor data for debugging
        """
        ...
    
    def async_read(self) -> Dict[str, Any]:
        """
        Asynchronously read tactile data from the sensor.
        
        Returns:
            Same format as read() method
        """
        ...
    
    def calibrate(self) -> None:
        """Calibrate the sensor (zero-point calibration)."""
        ...
    
    def disconnect(self) -> None:
        """Disconnect from the tactile sensor."""
        ...
    
    def is_connected(self) -> bool:
        """Check if sensor is connected."""
        ...
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """
        Get sensor information.
        
        Returns:
            Dict containing sensor info:
            - 'sensor_type': str, type of sensor
            - 'serial_number': str, sensor serial number
            - 'firmware_version': str, firmware version
            - 'sampling_rate': float, sampling rate in Hz
            - 'resolution': tuple, sensor resolution (if applicable)
        """
        ...


def make_tactile_sensors_from_configs(
    tactile_sensor_configs: Dict[str, TactileSensorConfig]
) -> Dict[str, TactileSensor]:
    """
    Create tactile sensors from configuration dictionary.
    
    Args:
        tactile_sensor_configs: Dictionary mapping sensor names to their configs
        
    Returns:
        Dictionary mapping sensor names to sensor instances
    """
    tactile_sensors = {}

    for key, cfg in tactile_sensor_configs.items():
        if cfg.type == "tac3d":
            from lerobot.common.robot_devices.tactile_sensors.tac3d import Tac3DSensor

            tactile_sensors[key] = Tac3DSensor(cfg)
            
        else:
            raise ValueError(f"The tactile sensor type '{cfg.type}' is not valid.")

    return tactile_sensors


def make_tactile_sensor(sensor_type: str, **kwargs) -> TactileSensor:
    """
    Create a single tactile sensor instance.
    
    Args:
        sensor_type: Type of sensor to create
        **kwargs: Configuration parameters
        
    Returns:
        TactileSensor instance
    """
    if sensor_type == "tac3d":
        from lerobot.common.robot_devices.tactile_sensors.tac3d import Tac3DSensor

        config = Tac3DConfig(**kwargs)
        return Tac3DSensor(config)
        
    else:
        raise ValueError(f"The tactile sensor type '{sensor_type}' is not valid.")


def standardize_tactile_data(
    raw_data: Dict[str, Any], 
    sensor_type: str,
    sensor_id: str,
    timestamp: Optional[float] = None
) -> Dict[str, Any]:
    """
    Standardize tactile data from different sensors to a common format.
    
    Args:
        raw_data: Raw sensor data
        sensor_type: Type of sensor
        sensor_id: Unique sensor identifier
        timestamp: Override timestamp (uses current time if None)
        
    Returns:
        Standardized tactile data dictionary
    """
    import time
    
    standardized = {
        'timestamp': timestamp or time.time(),
        'sensor_id': sensor_id,
        'sensor_type': sensor_type,
        'raw_data': raw_data,
    }
    
    # Add sensor-specific standardization logic here
    if sensor_type == "tac3d":
        # Tac3D specific data mapping
        if '3D_Positions' in raw_data:
            standardized['positions_3d'] = raw_data['3D_Positions']
        if '3D_Displacements' in raw_data:
            standardized['displacements_3d'] = raw_data['3D_Displacements']
        if '3D_Forces' in raw_data:
            standardized['forces_3d'] = raw_data['3D_Forces']
        if '3D_ResultantForce' in raw_data:
            standardized['resultant_force'] = raw_data['3D_ResultantForce']
        if '3D_ResultantMoment' in raw_data:
            standardized['resultant_moment'] = raw_data['3D_ResultantMoment']
    
    return standardized
