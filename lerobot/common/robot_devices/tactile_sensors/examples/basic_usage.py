#!/usr/bin/env python3

"""
Basic usage examples for LeRobot tactile sensors.

This script demonstrates how to use the tactile sensor framework
with different configurations and modes.
"""

import time
import logging
from typing import Dict, Any

from lerobot.common.robot_devices.tactile_sensors.config import (
    Tac3DConfig,
    ForceTorqueConfig,
    TactileArrayConfig,
)
from lerobot.common.robot_devices.tactile_sensors.utils import (
    make_tactile_sensor,
    make_tactile_sensors_from_configs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_single_sensor():
    """Example: Using a single Tac3D sensor."""
    print("\n=== Single Tac3D Sensor Example ===")
    
    # Create sensor configuration
    config = Tac3DConfig(
        sensor_id="tac3d_finger",
        port=9988,
        timeout=5.0,
        auto_calibrate=True,
        verbose=True,
        mock=True  # Use mock sensor for demonstration
    )
    
    # Method 1: Using make_tactile_sensor factory function
    sensor = make_tactile_sensor("tac3d", **config.__dict__)
    
    try:
        # Connect to sensor
        sensor.connect()
        
        # Get sensor info
        info = sensor.get_sensor_info()
        print(f"Sensor Info: {info}")
        
        # Read some data
        for i in range(5):
            data = sensor.read()
            print(f"Frame {i+1}:")
            print(f"  Timestamp: {data['timestamp']:.3f}")
            print(f"  Sensor ID: {data['sensor_id']}")
            print(f"  Frame Index: {data.get('frame_index', 'N/A')}")
            
            if 'positions_3d' in data:
                positions = data['positions_3d']
                print(f"  3D Positions shape: {positions.shape}")
                print(f"  Mean position: {positions.mean(axis=0)}")
            
            if 'forces_3d' in data:
                forces = data['forces_3d']
                print(f"  3D Forces shape: {forces.shape}")
                print(f"  Total force magnitude: {(forces**2).sum()**0.5:.6f}")
            
            time.sleep(0.1)
    
    finally:
        sensor.disconnect()


def example_multiple_sensors():
    """Example: Using multiple tactile sensors."""
    print("\n=== Multiple Sensors Example ===")
    
    # Define configurations for multiple sensors
    sensor_configs = {
        "left_finger": Tac3DConfig(
            sensor_id="tac3d_left",
            port=9988,
            mock=True
        ),
        "right_finger": Tac3DConfig(
            sensor_id="tac3d_right", 
            port=9989,
            mock=True
        ),
        # Example of other sensor types (placeholders)
        # "wrist_ft": ForceTorqueConfig(
        #     sensor_id="ft_wrist",
        #     serial_port="/dev/ttyUSB0",
        #     mock=True
        # ),
    }
    
    # Create sensors from configurations
    sensors = make_tactile_sensors_from_configs(sensor_configs)
    
    try:
        # Connect all sensors
        for name, sensor in sensors.items():
            print(f"Connecting to {name}...")
            sensor.connect()
        
        # Read from all sensors
        for i in range(3):
            print(f"\n--- Reading {i+1} ---")
            for name, sensor in sensors.items():
                data = sensor.read()
                print(f"{name}: Frame {data.get('frame_index', 'N/A')}, "
                      f"Timestamp: {data['timestamp']:.3f}")
            time.sleep(0.2)
    
    finally:
        # Disconnect all sensors
        for name, sensor in sensors.items():
            print(f"Disconnecting {name}...")
            sensor.disconnect()


def example_context_manager():
    """Example: Using sensor with context manager."""
    print("\n=== Context Manager Example ===")
    
    config = Tac3DConfig(
        sensor_id="tac3d_context",
        port=9988,
        mock=True,
        enable_3d_forces=True,
        enable_resultant_force=True
    )
    
    sensor = make_tactile_sensor("tac3d", **config.__dict__)
    
    # Using context manager automatically handles connect/disconnect
    with sensor:
        print("Sensor connected via context manager")
        
        # Perform calibration
        sensor.calibrate()
        
        # Read data
        for i in range(3):
            data = sensor.read()
            
            if 'resultant_force' in data:
                force = data['resultant_force']
                print(f"Reading {i+1}: Resultant force = {force}")
            
            time.sleep(0.1)
    
    print("Sensor automatically disconnected")


def example_async_reading():
    """Example: Asynchronous reading mode."""
    print("\n=== Async Reading Example ===")
    
    config = Tac3DConfig(
        sensor_id="tac3d_async",
        port=9988,
        use_async_read=True,
        max_queue_size=10,
        mock=True
    )
    
    sensor = make_tactile_sensor("tac3d", **config.__dict__)
    
    try:
        sensor.connect()
        
        # Wait a bit for async buffer to fill
        time.sleep(0.5)
        
        # Read multiple frames quickly
        print("Reading frames asynchronously:")
        for i in range(5):
            data = sensor.async_read()
            print(f"  Frame {data.get('frame_index', 'N/A')} at {data['timestamp']:.3f}")
            time.sleep(0.05)  # Read faster than sensor rate
    
    finally:
        sensor.disconnect()


def example_data_filtering():
    """Example: Configuring data filtering."""
    print("\n=== Data Filtering Example ===")
    
    # Configuration with selective data types
    config = Tac3DConfig(
        sensor_id="tac3d_filtered",
        port=9988,
        enable_3d_positions=False,  # Disable positions
        enable_3d_displacements=False,  # Disable displacements
        enable_3d_forces=True,  # Enable forces only
        enable_resultant_force=True,
        enable_resultant_moment=False,
        mock=True
    )
    
    sensor = make_tactile_sensor("tac3d", **config.__dict__)
    
    with sensor:
        data = sensor.read()
        
        print("Available data fields:")
        for key, value in data.items():
            if key != 'raw_data':  # Skip raw data for brevity
                print(f"  {key}: {type(value).__name__}")
        
        print("\nFiltered out: positions_3d, displacements_3d, resultant_moment")


def example_error_handling():
    """Example: Error handling and recovery."""
    print("\n=== Error Handling Example ===")
    
    config = Tac3DConfig(
        sensor_id="tac3d_error_test",
        port=9988,
        timeout=1.0,  # Short timeout for demo
        mock=True
    )
    
    sensor = make_tactile_sensor("tac3d", **config.__dict__)
    
    try:
        # Try to read without connecting (should fail)
        try:
            data = sensor.read()
            print("This shouldn't print - sensor not connected")
        except RuntimeError as e:
            print(f"Expected error caught: {e}")
        
        # Connect and read normally
        sensor.connect()
        print("Connected successfully")
        
        # Check connection status
        print(f"Is connected: {sensor.is_connected()}")
        
        # Normal read
        data = sensor.read()
        print(f"Successfully read frame {data.get('frame_index', 'N/A')}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        if sensor.is_connected():
            sensor.disconnect()
            print("Sensor disconnected")


def main():
    """Run all examples."""
    print("LeRobot Tactile Sensors - Basic Usage Examples")
    print("=" * 50)
    
    try:
        example_single_sensor()
        example_multiple_sensors()
        example_context_manager()
        example_async_reading()
        example_data_filtering()
        example_error_handling()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        logger.exception("Detailed error information:")


if __name__ == "__main__":
    main() 