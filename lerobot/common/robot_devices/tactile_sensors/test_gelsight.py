#!/usr/bin/env python3
"""
Simple test script for GelSight sensor integration.

This script tests the GelSight sensor functionality without requiring the full robot setup.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def test_gelsight_import():
    """Test if we can import GelSight components."""
    print("Testing GelSight imports...")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
        print("✓ GelSightConfig import successful")
    except ImportError as e:
        print(f"✗ GelSightConfig import failed: {e}")
        return False
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.gelsight import GelSightSensor
        print("✓ GelSightSensor import successful")
    except ImportError as e:
        print(f"✗ GelSightSensor import failed: {e}")
        return False
    
    return True

def test_gelsight_config():
    """Test GelSight configuration."""
    print("\nTesting GelSight configuration...")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
        
        # Test default config
        config = GelSightConfig()
        print(f"✓ Default config created: {config.device_name}")
        
        # Test custom config
        custom_config = GelSightConfig(
            device_name="GelSight Mini",
            imgh=240,
            imgw=320,
            framerate=30
        )
        print(f"✓ Custom config created: {custom_config.device_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_gelsight_sensor():
    """Test GelSight sensor creation (without hardware)."""
    print("\nTesting GelSight sensor creation...")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig
        from lerobot.common.robot_devices.tactile_sensors.gelsight import GelSightSensor
        
        config = GelSightConfig(device_name="Test Device")
        sensor = GelSightSensor(config)
        
        print(f"✓ Sensor created: {sensor.device_name}")
        print(f"✓ Connection status: {sensor.is_connected()}")
        
        # Test sensor info
        info = sensor.get_sensor_info()
        print(f"✓ Sensor info: {info}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sensor creation test failed: {e}")
        return False

def test_gs_sdk():
    """Test if gs_sdk is available."""
    print("\nTesting gs_sdk availability...")
    
    try:
        from lerobot.common.robot_devices.tactile_sensors.gs_sdk.gs_sdk.gs_device import FastCamera
        print("✓ gs_sdk FastCamera import successful")
        return True
    except ImportError as e:
        print(f"✗ gs_sdk not available: {e}")
        print("  Make sure gs_sdk is installed: pip install -e lerobot/common/robot_devices/tactile_sensors/gs_sdk")
        return False

def main():
    """Run all tests."""
    print("=== GelSight Integration Test ===\n")
    
    tests = [
        test_gelsight_import,
        test_gelsight_config,
        test_gelsight_sensor,
        test_gs_sdk,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! GelSight integration is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main()) 