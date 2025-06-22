#!/usr/bin/env python3
"""
GelSight sensor usage example for LeRobot.

This example demonstrates how to use the GelSight sensor in different scenarios:
1. Basic sensor reading
2. Recording tactile data to a dataset
3. Real-time visualization

Run with:
    python examples/gelsight_example.py --mode basic
    python examples/gelsight_example.py --mode record --output_dir ./tactile_data
    python examples/gelsight_example.py --mode visualize
"""

import argparse
import time
import os
import cv2
import numpy as np
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

try:
    from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
    from lerobot.common.robot_devices.tactile_sensors.gelsight import GelSightSensor
except ImportError:
    print("Warning: Could not import LeRobot modules. Running in standalone mode.")
    # Use local fallback
    sys.path.append(str(Path(__file__).parent.parent))
    from configs import GelSightConfig
    from gelsight import GelSightSensor


def basic_reading_example(device_name: str = "GelSight Mini", duration: float = 10.0):
    """
    Basic example of reading data from GelSight sensor.
    
    Args:
        device_name: Name of the GelSight device
        duration: How long to read data (seconds)
    """
    print(f"=== Basic Reading Example ===")
    print(f"Device: {device_name}")
    print(f"Duration: {duration} seconds")
    
    # Create sensor configuration
    config = GelSightConfig(device_name=device_name, framerate=25)
    sensor = GelSightSensor(config)
    
    try:
        # Connect to sensor
        print("Connecting to sensor...")
        sensor.connect()
        
        print("Starting data reading...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            # Read data
            data = sensor.read()
            
            if data and 'image' in data:
                frame_count += 1
                
                # Print some info every 25 frames (1 second at 25fps)
                if frame_count % 25 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Frame {frame_count}: {elapsed:.1f}s, FPS: {fps:.1f}")
            
            # Small delay to avoid overwhelming the CPU
            time.sleep(0.01)
        
        final_fps = frame_count / duration
        print(f"Completed! Read {frame_count} frames, Average FPS: {final_fps:.1f}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sensor.disconnect()


def recording_example(device_name: str = "GelSight Mini", output_dir: str = "./tactile_data", duration: float = 5.0):
    """
    Example of recording tactile data to files.
    
    Args:
        device_name: Name of the GelSight device
        output_dir: Directory to save the data
        duration: How long to record (seconds)
    """
    print(f"=== Recording Example ===")
    print(f"Device: {device_name}")
    print(f"Output: {output_dir}")
    print(f"Duration: {duration} seconds")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sensor configuration
    config = GelSightConfig(device_name=device_name, framerate=25)
    sensor = GelSightSensor(config)
    
    try:
        # Connect to sensor
        print("Connecting to sensor...")
        sensor.connect()
        
        print("Starting recording...")
        start_time = time.time()
        frame_count = 0
        timestamps = []
        
        while time.time() - start_time < duration:
            # Read data
            data = sensor.read()
            
            if data and 'image' in data:
                # Save image
                image_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(image_path, data['image'])
                
                # Store timestamp
                timestamps.append(data['timestamp'])
                frame_count += 1
                
                if frame_count % 25 == 0:
                    elapsed = time.time() - start_time
                    print(f"Recorded {frame_count} frames ({elapsed:.1f}s)")
        
        # Save timestamps
        timestamp_path = os.path.join(output_dir, "timestamps.txt")
        with open(timestamp_path, 'w') as f:
            for i, ts in enumerate(timestamps):
                f.write(f"{i},{ts}\n")
        
        print(f"Recording complete! Saved {frame_count} frames to {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sensor.disconnect()


def visualization_example(device_name: str = "GelSight Mini"):
    """
    Example of real-time visualization of GelSight data.
    
    Args:
        device_name: Name of the GelSight device
    """
    print(f"=== Visualization Example ===")
    print(f"Device: {device_name}")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Create sensor configuration
    config = GelSightConfig(device_name=device_name, framerate=25)
    sensor = GelSightSensor(config)
    
    try:
        # Connect to sensor
        print("Connecting to sensor...")
        sensor.connect()
        
        print("Starting visualization...")
        frame_count = 0
        saved_count = 0
        
        while True:
            # Read data
            data = sensor.read()
            
            if data and 'image' in data:
                image = data['image']
                frame_count += 1
                
                # Add frame info overlay
                cv2.putText(image, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Press 'q' to quit, 's' to save", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display image
                cv2.imshow(f"GelSight - {device_name}", image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"gelsight_frame_{saved_count:03d}.jpg"
                    cv2.imwrite(save_path, image)
                    saved_count += 1
                    print(f"Saved frame to {save_path}")
        
        print(f"Visualization ended. Processed {frame_count} frames, saved {saved_count} frames")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sensor.disconnect()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="GelSight sensor examples for LeRobot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gelsight_example.py --mode basic --device "GelSight Mini" --duration 10
  python gelsight_example.py --mode record --output_dir ./my_data --duration 5
  python gelsight_example.py --mode visualize --device "GelSight Mini"
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["basic", "record", "visualize"],
        default="basic",
        help="Example mode to run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GelSight Mini",
        help="GelSight device name"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration for basic/record modes (seconds)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tactile_data",
        help="Output directory for record mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        basic_reading_example(args.device, args.duration)
    elif args.mode == "record":
        recording_example(args.device, args.output_dir, args.duration)
    elif args.mode == "visualize":
        visualization_example(args.device)


if __name__ == "__main__":
    main() 