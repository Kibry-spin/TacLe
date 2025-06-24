#!/usr/bin/env python3

"""
Export GelSight tactile sensor data from LeRobot dataset as video files.

This script extracts GelSight tactile images from a dataset and saves them as MP4 video files.
Supports the new hierarchical tactile sensor data structure: observation.tactile.gelsight.{name}.tactile_image

Examples:
    # Export all GelSight sensors from episode 0
    python export_gelsight_video.py --repo-id $USER/test_tt  --episode-index 0

    # Export specific sensor with custom output directory
    python export_gelsight_video.py --repo-id $USER/test_tt --episode-index 0 \
        --sensor-name main_gripper0 --output-dir ./gelsight_videos

    # Export with custom video settings
    python export_gelsight_video.py --repo-id $USER/test_tt --episode-index 0 \
        --fps 30 --quality high --format avi
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    """Sampler for extracting a specific episode from the dataset."""
    
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)


def convert_tensor_to_bgr_numpy(tensor_image: torch.Tensor) -> np.ndarray:
    """
    Convert tensor image to BGR numpy array for OpenCV video writing.
    
    Args:
        tensor_image: Input tensor image (C, H, W) or (H, W, C)
        
    Returns:
        BGR numpy array (H, W, 3) with dtype uint8
    """
    if isinstance(tensor_image, torch.Tensor):
        if tensor_image.ndim == 3:
            if tensor_image.shape[0] == 3:  # (C, H, W) format
                # Convert to (H, W, C)
                if tensor_image.dtype == torch.float32:
                    # Float32 tensor, convert to uint8
                    image_np = (tensor_image * 255).type(torch.uint8).permute(1, 2, 0).numpy()
                else:
                    # Already uint8, just permute
                    image_np = tensor_image.permute(1, 2, 0).numpy()
            else:  # (H, W, C) format
                if tensor_image.dtype == torch.float32:
                    # Float32 tensor, convert to uint8
                    image_np = (tensor_image * 255).type(torch.uint8).numpy()
                else:
                    # Already uint8
                    image_np = tensor_image.numpy()
        else:
            raise ValueError(f"Unexpected tensor dimensions: {tensor_image.shape}")
    else:
        # Already numpy array
        image_np = tensor_image
    
    # Ensure uint8 dtype
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            # Normalized float image
            image_np = (image_np * 255).astype(np.uint8)
        else:
            # Other cases
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    
    # GelSight数据处理说明：
    # 1. GelSight传感器使用FFmpeg的bgr24格式捕获
    # 2. 在数据集中可能被存储为RGB格式
    # 3. OpenCV VideoWriter期望BGR格式
    # 4. 我们需要确保最终输出是正确的BGR格式
    
    if image_np.shape[-1] == 3:
        # 检查当前数据是BGR还是RGB格式
        # 由于数据集可能已经转换过，我们需要确保输出是BGR
        # 这里假设输入数据是RGB格式，转换为BGR供OpenCV使用
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_np


def get_video_codec_and_extension(format_type: str, quality: str) -> tuple:
    """
    Get video codec and file extension based on format and quality.
    
    Args:
        format_type: Video format ('mp4', 'avi', 'mov')
        quality: Quality setting ('low', 'medium', 'high', 'lossless')
        
    Returns:
        (fourcc, extension) tuple
    """
    if format_type.lower() == 'mp4':
        if quality == 'lossless':
            # 使用更兼容的无损编码
            return cv2.VideoWriter_fourcc(*'H264'), '.mp4'
        elif quality == 'high':
            # 使用H.264编码，兼容性更好
            return cv2.VideoWriter_fourcc(*'H264'), '.mp4'
        else:
            # 使用标准H.264编码
            return cv2.VideoWriter_fourcc(*'H264'), '.mp4'
    elif format_type.lower() == 'avi':
        if quality == 'lossless':
            # AVI格式使用MJPG编码，兼容性好
            return cv2.VideoWriter_fourcc(*'MJPG'), '.avi'
        else:
            return cv2.VideoWriter_fourcc(*'MJPG'), '.avi'
    elif format_type.lower() == 'mov':
        # MOV格式使用H.264
        return cv2.VideoWriter_fourcc(*'H264'), '.mov'
    else:
        # 默认使用AVI+MJPG，兼容性最好
        return cv2.VideoWriter_fourcc(*'MJPG'), '.avi'


def find_gelsight_sensors(dataset: LeRobotDataset) -> List[str]:
    """
    Find all GelSight sensors in the dataset.
    
    Args:
        dataset: LeRobot dataset
        
    Returns:
        List of GelSight sensor names
    """
    gelsight_sensors = []
    
    # Check dataset features for GelSight tactile data
    for feature_name in dataset.features:
        if feature_name.startswith("observation.tactile.gelsight.") and feature_name.endswith(".tactile_image"):
            # Extract sensor name: observation.tactile.gelsight.{sensor_name}.tactile_image
            parts = feature_name.split(".")
            if len(parts) >= 5:
                sensor_name = parts[3]
                if sensor_name not in gelsight_sensors:
                    gelsight_sensors.append(sensor_name)
    
    return gelsight_sensors


def export_gelsight_video(
    dataset: LeRobotDataset,
    episode_index: int,
    sensor_name: Optional[str] = None,
    output_dir: Path = Path("./gelsight_videos"),
    fps: int = 30,
    quality: str = "medium",
    format_type: str = "avi",
    batch_size: int = 32,
    num_workers: int = 4,
    save_debug_frames: bool = False
) -> List[Path]:
    """
    Export GelSight tactile sensor data as video files.
    
    Args:
        dataset: LeRobot dataset
        episode_index: Episode index to export
        sensor_name: Specific sensor name to export (None for all)
        output_dir: Output directory for video files
        fps: Video frame rate
        quality: Video quality ('low', 'medium', 'high', 'lossless')
        format_type: Video format ('mp4', 'avi', 'mov')
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        save_debug_frames: Save debug frames for color checking
        
    Returns:
        List of created video file paths
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GelSight sensors
    gelsight_sensors = find_gelsight_sensors(dataset)
    
    if not gelsight_sensors:
        logging.warning("No GelSight sensors found in the dataset")
        return []
    
    # Filter sensors if specific sensor requested
    if sensor_name:
        if sensor_name in gelsight_sensors:
            gelsight_sensors = [sensor_name]
        else:
            logging.error(f"Sensor '{sensor_name}' not found. Available sensors: {gelsight_sensors}")
            return []
    
    logging.info(f"Found GelSight sensors: {gelsight_sensors}")
    
    # Setup data loader
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )
    
    # Get video codec and extension
    fourcc, file_ext = get_video_codec_and_extension(format_type, quality)
    
    # Initialize video writers
    video_writers: Dict[str, cv2.VideoWriter] = {}
    video_paths: List[Path] = []
    
    # Process data
    frame_count = 0
    repo_id_str = dataset.repo_id.replace("/", "_")
    
    logging.info("Processing frames...")
    
    for batch in tqdm.tqdm(dataloader, desc="Exporting GelSight videos"):
        for i in range(len(batch["index"])):
            frame_count += 1
            
            # Process each GelSight sensor
            for sensor in gelsight_sensors:
                tactile_key = f"observation.tactile.gelsight.{sensor}.tactile_image"
                
                if tactile_key in batch:
                    # Get tactile image
                    tactile_image = batch[tactile_key][i]
                    
                    # Convert to BGR numpy for OpenCV
                    try:
                        bgr_image = convert_tensor_to_bgr_numpy(tactile_image)
                        
                        # 保存调试帧（仅前几帧）
                        if save_debug_frames and frame_count <= 3:
                            debug_filename = f"debug_frame_{frame_count}_{sensor}_bgr.png"
                            debug_path = output_dir / debug_filename
                            cv2.imwrite(str(debug_path), bgr_image)
                            
                            # 同时保存RGB版本用于对比
                            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                            debug_rgb_filename = f"debug_frame_{frame_count}_{sensor}_rgb.png"
                            debug_rgb_path = output_dir / debug_rgb_filename
                            cv2.imwrite(str(debug_rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                            
                            logging.info(f"Saved debug frames: {debug_path} and {debug_rgb_path}")
                            
                    except Exception as e:
                        logging.warning(f"Failed to convert image for sensor {sensor}, frame {frame_count}: {e}")
                        continue
                    
                    # Initialize video writer if needed
                    if sensor not in video_writers:
                        video_filename = f"{repo_id_str}_episode_{episode_index}_gelsight_{sensor}{file_ext}"
                        video_path = output_dir / video_filename
                        video_paths.append(video_path)
                        
                        height, width = bgr_image.shape[:2]
                        video_writers[sensor] = cv2.VideoWriter(
                            str(video_path), fourcc, fps, (width, height)
                        )
                        
                        if not video_writers[sensor].isOpened():
                            logging.error(f"Failed to open video writer for {video_path}")
                            continue
                        
                        logging.info(f"Created video writer for sensor '{sensor}': {video_path}")
                        logging.info(f"Video resolution: {width}x{height}, FPS: {fps}")
                    
                    # Write frame to video
                    video_writers[sensor].write(bgr_image)
    
    # Close all video writers
    for sensor, writer in video_writers.items():
        writer.release()
        logging.info(f"Video export completed for sensor '{sensor}'")
    
    logging.info(f"Successfully exported {frame_count} frames for {len(gelsight_sensors)} sensors")
    
    return video_paths


def main():
    parser = argparse.ArgumentParser(
        description="Export GelSight tactile sensor data from LeRobot dataset as video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_gelsight_video.py --repo-id $USER/test_tt --episode-index 0
  python export_gelsight_video.py --repo-id $USER/test_tt --episode-index 0 --sensor-name main_gripper0
  python export_gelsight_video.py --repo-id $USER/test_tt --episode-index 0 --fps 60 --quality high --format avi
        """
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. '$USER/test_tt')"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode index to export"
    )
    parser.add_argument(
        "--sensor-name",
        type=str,
        default=None,
        help="Specific GelSight sensor name to export (e.g. 'main_gripper0'). If not specified, exports all sensors"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./gelsight_videos"),
        help="Output directory for video files (default: './gelsight_videos')"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally. By default, uses hugging face cache"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)"
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=["low", "medium", "high", "lossless"],
        default="medium",
        help="Video quality setting (default: medium)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mp4", "avi", "mov"],
        default="avi",
        help="Video format (default: avi, most compatible)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loading (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save first few frames as images for debugging color issues"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load dataset
    logging.info(f"Loading dataset: {args.repo_id}")
    start_time = time.time()
    
    try:
        dataset = LeRobotDataset(args.repo_id, root=args.root)
        load_time = time.time() - start_time
        logging.info(f"Dataset loaded successfully in {load_time:.2f}s")
        logging.info(f"Dataset contains {dataset.num_episodes} episodes, {dataset.num_frames} total frames")
        
        # Validate episode index
        if args.episode_index >= dataset.num_episodes:
            raise ValueError(f"Episode index {args.episode_index} exceeds dataset size ({dataset.num_episodes} episodes)")
        
        # Export videos
        video_paths = export_gelsight_video(
            dataset=dataset,
            episode_index=args.episode_index,
            sensor_name=args.sensor_name,
            output_dir=args.output_dir,
            fps=args.fps,
            quality=args.quality,
            format_type=args.format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_debug_frames=args.save_frames
        )
        
        # Summary
        if video_paths:
            print(f"\n✅ Successfully exported {len(video_paths)} video(s):")
            for path in video_paths:
                file_size = path.stat().st_size / (1024 * 1024)  # MB
                print(f"  📹 {path} ({file_size:.1f} MB)")
        else:
            print("❌ No videos were exported")
            
    except Exception as e:
        logging.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main() 