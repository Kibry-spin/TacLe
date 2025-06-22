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
This file contains configuration classes for tactile sensors.
"""

import abc
from dataclasses import dataclass, field

import draccus


@dataclass  
class TactileSensorConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base class for tactile sensor configurations."""
    mock: bool = False
    
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@TactileSensorConfig.register_subclass("tac3d")
@dataclass
class Tac3DConfig(TactileSensorConfig):
    """
    Configuration for Tac3D tactile sensor.
    
    Args:
        port: UDP port for receiving sensor data (default: 9988)
        auto_calibrate: Whether to automatically calibrate on connection (default: True)
        mock: Whether to use mock sensor for testing (default: False)
    """
    port: int = 9988
    auto_calibrate: bool = True


@TactileSensorConfig.register_subclass("gelsight")
@dataclass
class GelSightConfig(TactileSensorConfig):
    """
    Configuration for GelSight tactile sensor.
    
    Args:
        device_name: Name of the camera device (default: "GelSight Mini")
        imgh: Desired image height (default: 240)
        imgw: Desired image width (default: 320)
        raw_imgh: Raw image height (default: 2464)
        raw_imgw: Raw image width (default: 3280)
        framerate: Camera framerate (default: 25)
        config_path: Path to sensor config file (optional)
        mock: Whether to use mock sensor for testing (default: False)
    """
    device_name: str = "GelSight Mini"
    imgh: int = 240
    imgw: int = 320
    raw_imgh: int = 2464
    raw_imgw: int = 3280
    framerate: int = 25
    config_path: str = "" 