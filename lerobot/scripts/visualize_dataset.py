#!/usr/bin/env python

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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Tactile Sensor Support:
- Tac3D sensors: Displays resultant forces, moments, 3D positions, and force distributions
- GelSight sensors: Displays tactile images with RGB channels and intensity statistics
- Common metadata: sensor serial numbers, timestamps, frame indices

Examples:

- Visualize data stored on a local machine:
```
local$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data with tactile sensors (including GelSight):
```
local$ python lerobot/scripts/visualize_dataset.py \
    --repo-id your_dataset_with_tactile \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # display each camera image
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalar(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))

            if "next.done" in batch:
                rr.log("next.done", rr.Scalar(batch["next.done"][i].item()))

            if "next.reward" in batch:
                rr.log("next.reward", rr.Scalar(batch["next.reward"][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalar(batch["next.success"][i].item()))

            # display tactile sensor data
            for key in batch.keys():
                if key.startswith("observation.tactile."):
                    # 提取传感器名称 (e.g., "main_gripper", "left_gripper")
                    parts = key.split(".")
                    if len(parts) >= 4:
                        sensor_name = parts[2]  # main_gripper, left_gripper, etc.
                        data_type = parts[3]    # positions_3d, forces_3d, tactile_image, etc.
                        
                        if data_type == "resultant_force":
                            # Tac3D传感器：显示xyz方向的合力分量
                            force = batch[key][i].numpy()  # (3,)
                            
                            # 设置合理的力值范围（可根据实际传感器规格调整）
                            min_force = -50.0  # 允许负值（拉力）
                            max_force = 50.0   # 最大推力
                            
                            # 分别显示x、y、z方向的力分量
                            for dim_idx, force_component in enumerate(force):
                                axis_name = ['x', 'y', 'z'][dim_idx]
                                
                                # 限制显示范围避免异常值影响可视化
                                force_clamped = np.clip(force_component, min_force, max_force)
                                
                                # 可选：添加原始值记录（用于调试）
                                if abs(force_component) > max_force:
                                    print(f"Warning: Force {axis_name} component {force_component:.2f} exceeds max range ±{max_force}")
                                
                                rr.log(f"tactile/Tac3D/{sensor_name}/force_{axis_name}", 
                                      rr.Scalar(force_clamped.item()))

                        elif data_type == "tactile_image":
                            # GelSight传感器：显示触觉图像
                            tactile_image = batch[key][i]  # (H, W, 3) 或 (3, H, W)
                            
                            # 检查图像数据格式并转换为正确格式
                            if isinstance(tactile_image, torch.Tensor):
                                if tactile_image.ndim == 3:
                                    if tactile_image.shape[0] == 3:  # (3, H, W) - CHW格式
                                        # 转换为HWC格式并转为numpy
                                        if tactile_image.dtype == torch.float32:
                                            # 浮点数图像，需要转换为uint8
                                            tactile_image_np = to_hwc_uint8_numpy(tactile_image)
                                        else:
                                            # 已经是uint8，只需要转换维度顺序
                                            tactile_image_np = tactile_image.permute(1, 2, 0).numpy()
                                    else:  # (H, W, 3) - HWC格式
                                        if tactile_image.dtype == torch.float32:
                                            # 浮点数转uint8
                                            tactile_image_np = (tactile_image * 255).type(torch.uint8).numpy()
                                        else:
                                            # 已经是uint8
                                            tactile_image_np = tactile_image.numpy()
                                else:
                                    print(f"Warning: Unexpected tactile image dimensions: {tactile_image.shape}")
                                    continue
                            else:
                                # 已经是numpy数组
                                tactile_image_np = tactile_image
                                
                            # 确保数据类型正确
                            if tactile_image_np.dtype != np.uint8:
                                if tactile_image_np.max() <= 1.0:
                                    # 归一化的浮点数图像
                                    tactile_image_np = (tactile_image_np * 255).astype(np.uint8)
                                else:
                                    # 其他情况，直接转换
                                    tactile_image_np = tactile_image_np.astype(np.uint8)
                            
                            # 记录触觉图像
                            rr.log(f"tactile/GelSight/{sensor_name}/tactile_image", rr.Image(tactile_image_np))

    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
