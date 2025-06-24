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

########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import rerun as rr
import torch
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    # logging.info(info_str)  # 禁用频繁的性能监控日志输出


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
                # 设置强制退出标志，确保程序能立即响应
                events["force_exit"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
                # ESC键不使用force_exit，让程序正常保存数据
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_data,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_data=display_data,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_data,
    policy,
    fps,
    single_task,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_data=display_data,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_data=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False, "force_exit": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()

    # Controls starts, if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()
            action = None

            if policy is not None:
                pred_action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                action = {"action": action}

        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            if action is not None:
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break
            
        # 检查强制退出标志，确保能立即响应
        if events.get("force_exit", False):
            print("检测到强制退出信号，立即停止...")
            break


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    # 重新校准触觉传感器
    if hasattr(robot, 'tactile_sensors') and robot.tactile_sensors:
        print("正在重新校准触觉传感器...")
        for name, sensor in robot.tactile_sensors.items():
            if sensor.is_connected():
                print(f"校准传感器 {name}...")
                sensor.calibrate()
        print("触觉传感器校准完成")

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
    )


def stop_recording(robot, listener, display_data, force_cleanup=False, disconnect_tactile=True):
    """Stop recording and properly cleanup all resources."""
    print("正在停止记录并清理资源...")
    
    try:
        # 根据情况选择正常断开或强制断开
        if disconnect_tactile:
            robot.disconnect(force=force_cleanup)
        else:
            # 只断开机械臂和相机，不断开触觉传感器（已经提前断开）
            print("断开机械臂和相机...")
            # Disconnect follower arms
            for name in robot.follower_arms:
                try:
                    robot.follower_arms[name].disconnect()
                    print(f"Follower arm {name} disconnected successfully.")
                except Exception as e:
                    print(f"Warning: Error disconnecting follower arm {name}: {e}")

            # Disconnect leader arms
            for name in robot.leader_arms:
                try:
                    robot.leader_arms[name].disconnect()
                    print(f"Leader arm {name} disconnected successfully.")
                except Exception as e:
                    print(f"Warning: Error disconnecting leader arm {name}: {e}")

            # Disconnect cameras
            for name in robot.cameras:
                try:
                    robot.cameras[name].disconnect()
                    print(f"Camera {name} disconnected successfully.")
                except Exception as e:
                    print(f"Warning: Error disconnecting camera {name}: {e}")
            
            robot.is_connected = False
        print("机器人连接已断开")
    except Exception as e:
        print(f"机器人断开连接时出错: {e}")
        if not force_cleanup:
            print("注意：如果有残留进程，请运行: python cleanup_gelsight_processes.py")

    # 停止键盘监听器
    try:
        if not is_headless() and listener is not None:
            listener.stop()
            print("键盘监听器已停止")
    except Exception as e:
        print(f"停止键盘监听器时出错: {e}")
        
    print("资源清理完成")


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
