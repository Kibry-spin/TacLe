#!/usr/bin/env python3
"""
Test script to verify DIGIT sensor functionality after fixes.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def test_digit_sensor():
    """Test DIGIT sensor connection and data capture."""
    print("=== DIGIT Sensor Test ===")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.digit import DIGITSensor
        from lerobot.common.robot_devices.tactile_sensors.configs import DIGITConfig
        
        # Create DIGIT sensor with correct device name
        config = DIGITConfig(
            device_name="DIGIT",  # This should match what's found in /sys/class/video4linux
            imgh=240,
            imgw=320,
            raw_imgh=480,
            raw_imgw=640,
            framerate=60
        )
        
        print(f"Created DIGIT config: {config.device_name}")
        
        # Create sensor instance
        sensor = DIGITSensor(config)
        print("DIGIT sensor instance created")
        
        # Connect to sensor
        print("Connecting to DIGIT sensor...")
        sensor.connect()
        print("✅ DIGIT sensor connected successfully!")
        
        # Test reading data
        print("Testing data reading...")
        
        for i in range(10):
            data = sensor.read()
            if data is not None:
                print(f"Frame {i+1}:")
                print(f"  Device name: {data.get('device_name')}")
                print(f"  Frame index: {data.get('frame_index')}")
                print(f"  Timestamp: {data.get('timestamp')}")
                
                if 'tactile_image' in data:
                    image = data['tactile_image']
                    print(f"  Image shape: {image.shape}")
                    print(f"  Image dtype: {image.dtype}")
                    print(f"  Image range: [{image.min()}, {image.max()}]")
                    
                    # Check if image has actual data (not all zeros)
                    non_zero_pixels = np.count_nonzero(image)
                    total_pixels = image.size
                    print(f"  Non-zero pixels: {non_zero_pixels}/{total_pixels} ({100*non_zero_pixels/total_pixels:.1f}%)")
                    
                    if non_zero_pixels > 0:
                        print("  ✅ Image contains actual data!")
                    else:
                        print("  ⚠️  Image is all zeros")
                else:
                    print("  ❌ No tactile_image in data")
                
                print()
            else:
                print(f"Frame {i+1}: No data received")
            
            time.sleep(0.5)
        
        # Disconnect
        print("Disconnecting sensor...")
        sensor.disconnect()
        print("✅ DIGIT sensor disconnected successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gs_device_directly():
    """Test the gs_device Camera class directly to isolate the problem."""
    print("\n=== Direct gs_device Test ===")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.gs_sdk.gs_sdk.gs_device import Camera, get_camera_id
        
        # Test get_camera_id function
        print("Testing camera ID detection...")
        camera_id = get_camera_id("DIGIT", verbose=True)
        print(f"Camera ID for DIGIT: {camera_id}")
        
        if camera_id is None:
            print("❌ Failed to find DIGIT camera")
            return False
        
        # Test direct camera connection
        print(f"\nTesting direct camera connection to device {camera_id}...")
        camera = Camera("DIGIT", 240, 320)
        camera.connect()
        print("✅ Camera connected successfully!")
        
        # Test image capture
        print("Testing image capture...")
        for i in range(5):
            image = camera.get_image(flush=(i==0))  # Flush first frame
            if image is not None:
                print(f"Frame {i+1}: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
                
                # Check for actual data
                non_zero = np.count_nonzero(image)
                print(f"  Non-zero pixels: {non_zero}/{image.size} ({100*non_zero/image.size:.1f}%)")
            else:
                print(f"Frame {i+1}: Failed to get image")
            time.sleep(0.2)
        
        camera.release()
        print("✅ Camera released successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Direct gs_device test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing DIGIT sensor fixes...")
    
    # Test 1: Direct gs_device test
    success1 = test_gs_device_directly()
    
    # Test 2: Our DIGIT sensor wrapper
    success2 = test_digit_sensor()
    
    print("\n=== Test Summary ===")
    print(f"Direct gs_device test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"DIGIT sensor wrapper test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! DIGIT sensor should work correctly now.")
    elif success1 and not success2:
        print("\n⚠️  Hardware works but wrapper has issues. Check DIGIT sensor implementation.")
    elif not success1:
        print("\n❌ Hardware connection failed. Check DIGIT device connection.") 