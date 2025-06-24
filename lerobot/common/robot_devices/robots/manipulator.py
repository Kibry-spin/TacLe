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

"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
import time
import warnings
import signal
import atexit
from pathlib import Path

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.configs import CameraConfig
from lerobot.common.robot_devices.cameras.utils import Camera, make_cameras_from_configs
from lerobot.common.robot_devices.motors.configs import MotorsBusConfig
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.tactile_sensors.configs import TactileSensorConfig
from lerobot.common.robot_devices.tactile_sensors.utils import TactileSensor, make_tactile_sensors_from_configs
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested relative goal position target: {diff}\n"
            f"    clamped relative goal position target: {safe_diff}"
        )

    return safe_goal_pos


class ManipulatorRobot:
    # TODO(rcadene): Implement force feedback
    """This class allows to control any manipulator robot of various number of motors.

    Non exhaustive list of robots:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow expansion, developed
    by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    - [Aloha](https://www.trossenrobotics.com/aloha-kits) developed by Trossen Robotics

    Example of instantiation, a pre-defined robot config is required:
    ```python
    robot = ManipulatorRobot(KochRobotConfig())
    ```

    Example of overwriting motors during instantiation:
    ```python
    # Defines how to communicate with the motors of the leader and follower arms
    leader_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0031751",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        ),
    }
    follower_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0032081",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        ),
    }
    robot_config = KochRobotConfig(leader_arms=leader_arms, follower_arms=follower_arms)
    robot = ManipulatorRobot(robot_config)
    ```

    Example of overwriting cameras during instantiation:
    ```python
    # Defines how to communicate with 2 cameras connected to the computer.
    # Here, the webcam of the laptop and the phone (connected in USB to the laptop)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "laptop": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }
    robot = ManipulatorRobot(KochRobotConfig(cameras=cameras))
    ```

    Once the robot is instantiated, connect motors buses and cameras if any (Required):
    ```python
    robot.connect()
    ```

    Example of highest frequency teleoperation, which doesn't require cameras:
    ```python
    while True:
        robot.teleop_step()
    ```

    Example of highest frequency data collection from motors and cameras (if any):
    ```python
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of controlling the robot with a policy:
    ```python
    while True:
        # Uses the follower arms and cameras to capture an observation
        observation = robot.capture_observation()

        # Assumes a policy has been instantiated
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Orders the robot to move
        robot.send_action(action)
    ```

    Example of disconnecting which is not mandatory since we disconnect when the object is deleted:
    ```python
    robot.disconnect()
    ```
    """

    def __init__(
        self,
        config: ManipulatorRobotConfig,
    ):
        self.robot_type = config.type
        self.config = config
        self.calibration_dir = Path(config.calibration_dir)

        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.leader_arms = make_motors_buses_from_configs(config.leader_arms)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.tactile_sensors = self._make_tactile_sensors_from_configs(config.tactile_sensors)

        self.is_connected = False
        self.logs = {}
        
        # Register signal handlers for graceful shutdown
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register atexit handler as fallback
        atexit.register(self._cleanup_at_exit)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals to ensure proper cleanup."""
        print(f"\nReceived signal {signum}. Cleaning up ManipulatorRobot...")
        try:
            if self.is_connected:
                self.disconnect()
        except Exception as e:
            print(f"Error during signal cleanup: {e}")
        finally:
            # Restore original handler and re-raise signal
            if signum == signal.SIGINT:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            elif signum == signal.SIGTERM:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            
            # Re-raise the signal to allow normal termination
            if signum == signal.SIGINT:
                raise KeyboardInterrupt()
            else:
                exit(1)

    def _cleanup_at_exit(self):
        """Cleanup function called at program exit."""
        try:
            if getattr(self, 'is_connected', False):
                print("Cleanup at exit: disconnecting robot...")
                self.disconnect()
        except Exception as e:
            print(f"Error during exit cleanup: {e}")

    def get_motor_names(self, arms: dict[str, MotorsBus]) -> list:
        motor_names = []
        for arm_name, arm in arms.items():
            motor_names.extend(list(arm.motors.keys()))
        return motor_names

    def _make_tactile_sensors_from_configs(self, configs: dict[str, TactileSensorConfig]) -> dict[str, TactileSensor]:
        """Create tactile sensor instances from configurations."""
        return make_tactile_sensors_from_configs(configs)

    @property
    def camera_features(self) -> dict:
        features = {}
        for name, camera in self.cameras.items():
            # TODO(rcadene): add more info (e.g. fps, width, height)
            features[f"observation.images.{name}"] = {
                "shape": (camera.height, camera.width, camera.channels),
                "names": ["height", "width", "channels"],
                "dtype": "video",
            }
        return features

    @property
    def tactile_features(self) -> dict:
        """Return the features associated with tactile sensors."""
        tactile_ft = {}
        for name in self.tactile_sensors:
            sensor = self.tactile_sensors[name]
            
            # 通过类名检测传感器类型
            sensor_type = 'unknown'
            if hasattr(sensor, 'config'):
                config_class_name = sensor.config.__class__.__name__.lower()
                if 'gelsight' in config_class_name:
                    sensor_type = 'gelsight'
                elif 'tac3d' in config_class_name:
                    sensor_type = 'tac3d'
                # 也检查是否有type属性
                elif hasattr(sensor.config, 'type'):
                    sensor_type = sensor.config.type
            
            # 基本元数据 (所有传感器通用)
            tactile_ft[f"observation.tactile.{name}.sensor_sn"] = {
                "dtype": "string",
                "shape": (1,),
                "names": None,
            }
            tactile_ft[f"observation.tactile.{name}.frame_index"] = {
                "dtype": "int64", 
                "shape": (1,),
                "names": None,
            }
            tactile_ft[f"observation.tactile.{name}.send_timestamp"] = {
                "dtype": "float64",
                "shape": (1,),
                "names": None,
            }
            tactile_ft[f"observation.tactile.{name}.recv_timestamp"] = {
                "dtype": "float64", 
                "shape": (1,),
                "names": None,
                        }
            
            if sensor_type == 'tac3d':
                # Tac3D传感器特有的三维数据阵列 (400个标志点，每个3维坐标)
                tactile_ft[f"observation.tactile.{name}.positions_3d"] = {
                    "dtype": "float64",
                    "shape": (400, 3),
                    "names": ["marker_id", "coordinate"],
                }
                tactile_ft[f"observation.tactile.{name}.displacements_3d"] = {
                    "dtype": "float64",
                    "shape": (400, 3), 
                    "names": ["marker_id", "coordinate"],
                }
                tactile_ft[f"observation.tactile.{name}.forces_3d"] = {
                    "dtype": "float64",
                    "shape": (400, 3),
                    "names": ["marker_id", "coordinate"],
                }
                # 合成力和力矩
                tactile_ft[f"observation.tactile.{name}.resultant_force"] = {
                    "dtype": "float64",
                    "shape": (3,),
                    "names": ["x", "y", "z"],
                }
                tactile_ft[f"observation.tactile.{name}.resultant_moment"] = {
                    "dtype": "float64",
                    "shape": (3,),
                    "names": ["x", "y", "z"],
                }
            elif sensor_type == 'gelsight':
                # GelSight传感器特有的图像数据
                # 获取图像尺寸配置
                imgh = getattr(sensor.config, 'imgh', 240)
                imgw = getattr(sensor.config, 'imgw', 320)
                
                tactile_ft[f"observation.tactile.{name}.tactile_image"] = {
                    "dtype": "uint8",
                    "shape": (imgh, imgw, 3),
                    "names": ["height", "width", "channel"],
                }
                # 可选：添加处理后的特征（如接触检测结果）
                # 这些可以通过图像处理算法从tactile_image计算得出
                # tactile_ft[f"observation.tactile.{name}.contact_map"] = {
                #     "dtype": "float32",
                #     "shape": (imgh, imgw),
                #     "names": ["height", "width"],
                # }
            else:
                # 未知传感器类型，使用基本的力数据格式
                tactile_ft[f"observation.tactile.{name}.resultant_force"] = {
                    "dtype": "float64",
                    "shape": (3,),
                    "names": ["x", "y", "z"],
                }
                tactile_ft[f"observation.tactile.{name}.resultant_moment"] = {
                    "dtype": "float64",
                    "shape": (3,),
                    "names": ["x", "y", "z"],
                }
                
        return tactile_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features, **self.tactile_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the leader arms
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm...")
            self.leader_arms[name].connect()

        # Connect the follower arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm...")
            self.follower_arms[name].connect()

        # Connect the cameras
        for name in self.cameras:
            print(f"Connecting {name} camera...")
            self.cameras[name].connect()

        # Connect the tactile sensors
        for name in self.tactile_sensors:
            print(f"Connecting {name} tactile sensor...")
            self.tactile_sensors[name].connect()

        self.activate_calibration()

        if self.robot_type == "aloha":
            self.set_aloha_robot_preset()
        elif self.robot_type in ["koch", "koch_bimanual"]:
            self.set_koch_robot_preset()
        elif self.robot_type == "so100":
            self.set_so100_robot_preset()

        self.is_connected = True

    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                # TODO(rcadene): display a warning in __init__ if calibration file not available
                print(f"Missing calibration file '{arm_calib_path}'")

                if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
                    from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration

                    calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)

                elif self.robot_type in ["so100", "so101", "moss", "lekiwi"]:
                    from lerobot.common.robot_devices.robots.feetech_calibration import (
                        run_arm_manual_calibration,
                    )

                    calibration = run_arm_manual_calibration(arm, self.robot_type, name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

        for name, arm in self.follower_arms.items():
            calibration = load_or_run_calibration_(name, arm, "follower")
            arm.set_calibration(calibration)
        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)

    def set_koch_robot_preset(self):
        def set_operating_mode_(arm):
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

            if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
                raise ValueError("To run set robot preset, the torque must be disabled on all motors.")

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Koch motors
                arm.write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for gripper to be limited by the limit of the current.
            # For the follower gripper, it means it can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
            # to make it move, and it will move back to its original target position when we release the force.
            # 5 corresponds to Current Controlled Position on Koch gripper motors "xl330-m077, xl330-m288"
            arm.write("Operating_Mode", 5, "gripper")

        for name in self.follower_arms:
            set_operating_mode_(self.follower_arms[name])

            # Set better PID values to close the gap between recorded states and actions
            # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor
            self.follower_arms[name].write("Position_P_Gain", 1500, "elbow_flex")
            self.follower_arms[name].write("Position_I_Gain", 0, "elbow_flex")
            self.follower_arms[name].write("Position_D_Gain", 600, "elbow_flex")

        if self.config.gripper_open_degree is not None:
            for name in self.leader_arms:
                set_operating_mode_(self.leader_arms[name])

                # Enable torque on the gripper of the leader arms, and move it to 45 degrees,
                # so that we can use it as a trigger to close the gripper of the follower arms.
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

    def set_aloha_robot_preset(self):
        def set_shadow_(arm):
            # Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
            # As a result, if only one of them is required to move to a certain position,
            # the other will follow. This is to avoid breaking the motors.
            if "shoulder_shadow" in arm.motor_names:
                shoulder_idx = arm.read("ID", "shoulder")
                arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

            if "elbow_shadow" in arm.motor_names:
                elbow_idx = arm.read("ID", "elbow")
                arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

        for name in self.follower_arms:
            set_shadow_(self.follower_arms[name])

        for name in self.leader_arms:
            set_shadow_(self.leader_arms[name])

        for name in self.follower_arms:
            # Set a velocity limit of 131 as advised by Trossen Robotics
            self.follower_arms[name].write("Velocity_Limit", 131)

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [
                name for name in self.follower_arms[name].motor_names if name != "gripper"
            ]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Aloha motors
                self.follower_arms[name].write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for follower gripper to be limited by the limit of the current.
            # It can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # 5 corresponds to Current Controlled Position on Aloha gripper follower "xm430-w350"
            self.follower_arms[name].write("Operating_Mode", 5, "gripper")

            # Note: We can't enable torque on the leader gripper since "xc430-w150" doesn't have
            # a Current Controlled Position mode.

        if self.config.gripper_open_degree is not None:
            warnings.warn(
                f"`gripper_open_degree` is set to {self.config.gripper_open_degree}, but None is expected for Aloha instead",
                stacklevel=1,
            )

    def set_so100_robot_preset(self):
        for name in self.follower_arms:
            # Mode=0 for Position Control
            self.follower_arms[name].write("Mode", 0)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.follower_arms[name].write("P_Coefficient", 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.follower_arms[name].write("I_Coefficient", 0)
            self.follower_arms[name].write("D_Coefficient", 32)
            # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
            # which is mandatory for Maximum_Acceleration to take effect after rebooting.
            self.follower_arms[name].write("Lock", 0)
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            self.follower_arms[name].write("Maximum_Acceleration", 254)
            self.follower_arms[name].write("Acceleration", 254)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Used when record_data=True
            follower_goal_pos[name] = goal_pos

            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # Early exit when recording data is not requested
        if not record_data:
            return

        # TODO(rcadene): Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Read tactile sensor data
        tactile_data = {}
        for name in self.tactile_sensors:
            before_tactile_read_t = time.perf_counter()
            sensor = self.tactile_sensors[name]
            sensor_type = getattr(sensor.config, 'type', 'unknown')
            data = sensor.read()
            
            if data:
                # 基本元数据 (所有传感器通用)
                tactile_data[f"{name}_sensor_sn"] = data.get('SN', '')
                tactile_data[f"{name}_frame_index"] = torch.tensor([data.get('index', 0)], dtype=torch.int64)
                tactile_data[f"{name}_send_timestamp"] = torch.tensor([data.get('sendTimestamp', 0.0)], dtype=torch.float64)
                tactile_data[f"{name}_recv_timestamp"] = torch.tensor([data.get('recvTimestamp', 0.0)], dtype=torch.float64)
                
                if sensor_type == 'tac3d':
                    # Tac3D传感器的三维数据阵列
                    if '3D_Positions' in data and data['3D_Positions'] is not None:
                        positions = data['3D_Positions']
                        tactile_data[f"{name}_positions_3d"] = torch.from_numpy(positions.astype(np.float64))
                    else:
                        tactile_data[f"{name}_positions_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    
                    if '3D_Displacements' in data and data['3D_Displacements'] is not None:
                        displacements = data['3D_Displacements'] 
                        tactile_data[f"{name}_displacements_3d"] = torch.from_numpy(displacements.astype(np.float64))
                    else:
                        tactile_data[f"{name}_displacements_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    
                    if '3D_Forces' in data and data['3D_Forces'] is not None:
                        forces_3d = data['3D_Forces']
                        tactile_data[f"{name}_forces_3d"] = torch.from_numpy(forces_3d.astype(np.float64))
                    else:
                        tactile_data[f"{name}_forces_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    
                    # 合成力和力矩
                    if 'resultant_force' in data and data['resultant_force'] is not None:
                        force = data['resultant_force']
                        if force.size >= 3:
                            tactile_data[f"{name}_resultant_force"] = torch.tensor([force[0, 0], force[0, 1], force[0, 2]], dtype=torch.float64)
                        else:
                            tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    else:
                        tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    
                    if 'resultant_moment' in data and data['resultant_moment'] is not None:
                        moment = data['resultant_moment']
                        if moment.size >= 3:
                            tactile_data[f"{name}_resultant_moment"] = torch.tensor([moment[0, 0], moment[0, 1], moment[0, 2]], dtype=torch.float64)
                        else:
                            tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
                    else:
                        tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
                        
                elif sensor_type == 'gelsight':
                    # GelSight传感器的图像数据
                    if 'tactile_image' in data and data['tactile_image'] is not None:
                        image = data['tactile_image']
                        tactile_data[f"{name}_tactile_image"] = torch.from_numpy(image)
                    elif 'image' in data and data['image'] is not None:
                        # 兼容旧版本的image字段
                        image = data['image']
                        tactile_data[f"{name}_tactile_image"] = torch.from_numpy(image)
                else:
                    # 未知传感器类型，使用基本的力数据格式
                    tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
                
            else:
                # 如果没有数据，根据传感器类型填充默认值
                tactile_data[f"{name}_sensor_sn"] = ''
                tactile_data[f"{name}_frame_index"] = torch.tensor([0], dtype=torch.int64)
                tactile_data[f"{name}_send_timestamp"] = torch.tensor([0.0], dtype=torch.float64)
                tactile_data[f"{name}_recv_timestamp"] = torch.tensor([0.0], dtype=torch.float64)
                
                if sensor_type == 'tac3d':
                    tactile_data[f"{name}_positions_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    tactile_data[f"{name}_displacements_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    tactile_data[f"{name}_forces_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
                elif sensor_type == 'gelsight':
                    imgh = getattr(sensor.config, 'imgh', 240)
                    imgw = getattr(sensor.config, 'imgw', 320)
                    tactile_data[f"{name}_tactile_image"] = torch.zeros((imgh, imgw, 3), dtype=torch.uint8)
                else:
                    tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
            
            self.logs[f"read_tactile_{name}_dt_s"] = time.perf_counter() - before_tactile_read_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        for name in self.tactile_sensors:
            sensor = self.tactile_sensors[name]
            sensor_type = getattr(sensor.config, 'type', 'unknown')
            
            # 基本元数据 (所有传感器通用)
            obs_dict[f"observation.tactile.{name}.sensor_sn"] = tactile_data[f"{name}_sensor_sn"]
            obs_dict[f"observation.tactile.{name}.frame_index"] = tactile_data[f"{name}_frame_index"]
            obs_dict[f"observation.tactile.{name}.send_timestamp"] = tactile_data[f"{name}_send_timestamp"]
            obs_dict[f"observation.tactile.{name}.recv_timestamp"] = tactile_data[f"{name}_recv_timestamp"]
            
            if sensor_type == 'tac3d':
                # Tac3D传感器特有的三维数据
                obs_dict[f"observation.tactile.{name}.positions_3d"] = tactile_data[f"{name}_positions_3d"]
                obs_dict[f"observation.tactile.{name}.displacements_3d"] = tactile_data[f"{name}_displacements_3d"]
                obs_dict[f"observation.tactile.{name}.forces_3d"] = tactile_data[f"{name}_forces_3d"]
                obs_dict[f"observation.tactile.{name}.resultant_force"] = tactile_data[f"{name}_resultant_force"]
                obs_dict[f"observation.tactile.{name}.resultant_moment"] = tactile_data[f"{name}_resultant_moment"]
            elif sensor_type == 'gelsight':
                # GelSight传感器特有的图像数据
                obs_dict[f"observation.tactile.{name}.tactile_image"] = tactile_data[f"{name}_tactile_image"]
            else:
                # 未知传感器类型，添加基本的力数据
                if f"{name}_resultant_force" in tactile_data:
                    obs_dict[f"observation.tactile.{name}.resultant_force"] = tactile_data[f"{name}_resultant_force"]
                if f"{name}_resultant_moment" in tactile_data:
                    obs_dict[f"observation.tactile.{name}.resultant_moment"] = tactile_data[f"{name}_resultant_moment"]

        # 创建action字典
        action_dict = {"action": action}
        
        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Read tactile sensor data
        tactile_data = {}
        for name in self.tactile_sensors:
            before_tactile_read_t = time.perf_counter()
            sensor = self.tactile_sensors[name]
            sensor_type = getattr(sensor.config, 'type', 'unknown')
            data = sensor.read()
            
            if data:
                # 基本元数据 (所有传感器通用)
                tactile_data[f"{name}_sensor_sn"] = data.get('SN', '')
                tactile_data[f"{name}_frame_index"] = torch.tensor([data.get('index', 0)], dtype=torch.int64)
                tactile_data[f"{name}_send_timestamp"] = torch.tensor([data.get('sendTimestamp', 0.0)], dtype=torch.float64)
                tactile_data[f"{name}_recv_timestamp"] = torch.tensor([data.get('recvTimestamp', 0.0)], dtype=torch.float64)
                
                if sensor_type == 'tac3d':
                    # Tac3D传感器的三维数据阵列
                    if '3D_Positions' in data and data['3D_Positions'] is not None:
                        positions = data['3D_Positions']
                        tactile_data[f"{name}_positions_3d"] = torch.from_numpy(positions.astype(np.float64))
                    else:
                        tactile_data[f"{name}_positions_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    
                    if '3D_Displacements' in data and data['3D_Displacements'] is not None:
                        displacements = data['3D_Displacements'] 
                        tactile_data[f"{name}_displacements_3d"] = torch.from_numpy(displacements.astype(np.float64))
                    else:
                        tactile_data[f"{name}_displacements_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    
                    if '3D_Forces' in data and data['3D_Forces'] is not None:
                        forces_3d = data['3D_Forces']
                        tactile_data[f"{name}_forces_3d"] = torch.from_numpy(forces_3d.astype(np.float64))
                    else:
                        tactile_data[f"{name}_forces_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    
                    # 合成力和力矩
                    if 'resultant_force' in data and data['resultant_force'] is not None:
                        force = data['resultant_force']
                        if force.size >= 3:
                            tactile_data[f"{name}_resultant_force"] = torch.tensor([force[0, 0], force[0, 1], force[0, 2]], dtype=torch.float64)
                        else:
                            tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    else:
                        tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    
                    if 'resultant_moment' in data and data['resultant_moment'] is not None:
                        moment = data['resultant_moment']
                        if moment.size >= 3:
                            tactile_data[f"{name}_resultant_moment"] = torch.tensor([moment[0, 0], moment[0, 1], moment[0, 2]], dtype=torch.float64)
                        else:
                            tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
                    else:
                        tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
                        
                elif sensor_type == 'gelsight':
                    # GelSight传感器的图像数据
                    if 'tactile_image' in data and data['tactile_image'] is not None:
                        image = data['tactile_image']
                        tactile_data[f"{name}_tactile_image"] = torch.from_numpy(image)
                    elif 'image' in data and data['image'] is not None:
                        # 兼容旧版本的image字段
                        image = data['image']
                        tactile_data[f"{name}_tactile_image"] = torch.from_numpy(image)
                else:
                    # 未知传感器类型，使用基本的力数据格式
                    tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
                
            else:
                # 如果没有数据，根据传感器类型填充默认值
                tactile_data[f"{name}_sensor_sn"] = ''
                tactile_data[f"{name}_frame_index"] = torch.tensor([0], dtype=torch.int64)
                tactile_data[f"{name}_send_timestamp"] = torch.tensor([0.0], dtype=torch.float64)
                tactile_data[f"{name}_recv_timestamp"] = torch.tensor([0.0], dtype=torch.float64)
                
                if sensor_type == 'tac3d':
                    tactile_data[f"{name}_positions_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    tactile_data[f"{name}_displacements_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    tactile_data[f"{name}_forces_3d"] = torch.zeros((400, 3), dtype=torch.float64)
                    tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
                elif sensor_type == 'gelsight':
                    imgh = getattr(sensor.config, 'imgh', 240)
                    imgw = getattr(sensor.config, 'imgw', 320)
                    tactile_data[f"{name}_tactile_image"] = torch.zeros((imgh, imgw, 3), dtype=torch.uint8)
                else:
                    tactile_data[f"{name}_resultant_force"] = torch.zeros(3, dtype=torch.float64)
                    tactile_data[f"{name}_resultant_moment"] = torch.zeros(3, dtype=torch.float64)
            
            self.logs[f"read_tactile_{name}_dt_s"] = time.perf_counter() - before_tactile_read_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        for name in self.tactile_sensors:
            sensor = self.tactile_sensors[name]
            sensor_type = getattr(sensor.config, 'type', 'unknown')
            
            # 基本元数据 (所有传感器通用)
            obs_dict[f"observation.tactile.{name}.sensor_sn"] = tactile_data[f"{name}_sensor_sn"]
            obs_dict[f"observation.tactile.{name}.frame_index"] = tactile_data[f"{name}_frame_index"]
            obs_dict[f"observation.tactile.{name}.send_timestamp"] = tactile_data[f"{name}_send_timestamp"]
            obs_dict[f"observation.tactile.{name}.recv_timestamp"] = tactile_data[f"{name}_recv_timestamp"]
            
            if sensor_type == 'tac3d':
                # Tac3D传感器特有的三维数据
                obs_dict[f"observation.tactile.{name}.positions_3d"] = tactile_data[f"{name}_positions_3d"]
                obs_dict[f"observation.tactile.{name}.displacements_3d"] = tactile_data[f"{name}_displacements_3d"]
                obs_dict[f"observation.tactile.{name}.forces_3d"] = tactile_data[f"{name}_forces_3d"]
                obs_dict[f"observation.tactile.{name}.resultant_force"] = tactile_data[f"{name}_resultant_force"]
                obs_dict[f"observation.tactile.{name}.resultant_moment"] = tactile_data[f"{name}_resultant_moment"]
            elif sensor_type == 'gelsight':
                # GelSight传感器特有的图像数据
                obs_dict[f"observation.tactile.{name}.tactile_image"] = tactile_data[f"{name}_tactile_image"]
            else:
                # 未知传感器类型，添加基本的力数据
                if f"{name}_resultant_force" in tactile_data:
                    obs_dict[f"observation.tactile.{name}.resultant_force"] = tactile_data[f"{name}_resultant_force"]
                if f"{name}_resultant_moment" in tactile_data:
                    obs_dict[f"observation.tactile.{name}.resultant_moment"] = tactile_data[f"{name}_resultant_moment"]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)

        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        # Disconnect follower arms
        for name in self.follower_arms:
            try:
                self.follower_arms[name].disconnect()
                print(f"Follower arm {name} disconnected successfully.")
            except Exception as e:
                print(f"Warning: Error disconnecting follower arm {name}: {e}")

        # Disconnect leader arms
        for name in self.leader_arms:
            try:
                self.leader_arms[name].disconnect()
                print(f"Leader arm {name} disconnected successfully.")
            except Exception as e:
                print(f"Warning: Error disconnecting leader arm {name}: {e}")

        # Disconnect cameras
        for name in self.cameras:
            try:
                self.cameras[name].disconnect()
                print(f"Camera {name} disconnected successfully.")
            except Exception as e:
                print(f"Warning: Error disconnecting camera {name}: {e}")

        # Disconnect tactile sensors with extra care
        for name in self.tactile_sensors:
            try:
                print(f"Disconnecting tactile sensor {name}...")
                self.tactile_sensors[name].disconnect()
                print(f"Tactile sensor {name} disconnected successfully.")
            except Exception as e:
                print(f"Warning: Error disconnecting tactile sensor {name}: {e}")
                # For GelSight sensors, try to force cleanup
                try:
                    sensor = self.tactile_sensors[name]
                    if hasattr(sensor, '_device') and sensor._device is not None:
                        if hasattr(sensor._device, 'release'):
                            print(f"Force releasing {name} device...")
                            sensor._device.release()
                        sensor._device = None
                    if hasattr(sensor, '_connected'):
                        sensor._connected = False
                    print(f"Force cleanup completed for {name}")
                except Exception as force_error:
                    print(f"Force cleanup failed for {name}: {force_error}")

        self.is_connected = False
        print("ManipulatorRobot disconnected successfully.")

    def __del__(self):
        """Destructor to ensure safe cleanup."""
        try:
            if getattr(self, "is_connected", False):
                print("ManipulatorRobot destructor: forcing disconnect...")
                self.disconnect()
        except Exception as e:
            print(f"Warning: Error in ManipulatorRobot destructor: {e}")
            # Force cleanup tactile sensors as last resort
            try:
                for name in getattr(self, 'tactile_sensors', {}):
                    try:
                        sensor = self.tactile_sensors[name]
                        if hasattr(sensor, '_device') and sensor._device:
                            if hasattr(sensor._device, 'release'):
                                sensor._device.release()
                            sensor._device = None
                        if hasattr(sensor, '_connected'):
                            sensor._connected = False
                    except:
                        pass
            except:
                pass
