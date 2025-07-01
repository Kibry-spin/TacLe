# Tac3D Tactile Sensor Data Format Specification

## Overview

This document describes the data format for Tac3D tactile sensors in the LeRobot framework. Tac3D sensors provide high-precision 3D tactile sensing with 400 marker points, displacement data, force fields, and resultant force/moment data.

## Sensor Specifications

- **Model**: AD2-0046R / AD2-0047L
- **Communication**: UDP protocol
- **Default port**: 9988
- **Data rate**: ~30Hz (up to 50Hz)
- **Markers**: 400 points
- **SDK version**: 3.2.1

## Key Data Structures

### 1. Raw Sensor Data Format

The `read()` method of Tac3DSensor returns the following standardized data structure:

```python
sensor_data = {
    # Metadata
    'recvTimestamp': 22.542121171951294,  # float: Receiving timestamp
    'SN': 'AD2-0046R',                    # str: Sensor serial number
    'index': 314,                         # int: Frame index
    
    # 3D data arrays (400 marker points)
    '3D_Positions': np.ndarray,           # shape=(400, 3), dtype=float64
    '3D_Displacements': np.ndarray,       # shape=(400, 3), dtype=float64  
    '3D_Forces': np.ndarray,              # shape=(400, 3), dtype=float64
    
    # Resultant data
    '3D_ResultantForce': np.ndarray,      # shape=(1, 3), dtype=float64
    '3D_ResultantMoment': np.ndarray,     # shape=(1, 3), dtype=float64
    
    # Standardized field names (added for compatibility)
    'resultant_force': np.ndarray,        # same as 3D_ResultantForce
    'resultant_moment': np.ndarray,       # same as 3D_ResultantMoment
    
    # Raw frame
    'raw_frame': dict                     # Complete raw UDP frame data
}
```

### 2. Field Details

| Field Name | Data Type | Shape | Unit | Description |
|------------|-----------|-------|------|-------------|
| `recvTimestamp` | float64 | (1,) | sec | Data receiving timestamp |
| `SN` | string | (1,) | - | Sensor serial number |
| `index` | int64 | (1,) | - | Frame index (incremental) |
| `3D_Positions` | float64 | (400, 3) | mm | [X,Y,Z] coordinates of 400 markers |
| `3D_Displacements` | float64 | (400, 3) | mm | [ΔX,ΔY,ΔZ] displacements |
| `3D_Forces` | float64 | (400, 3) | N | [Fx,Fy,Fz] local forces |
| `3D_ResultantForce` | float64 | (1, 3) | N | Resultant force vector [Fx,Fy,Fz] |
| `3D_ResultantMoment` | float64 | (1, 3) | N·m | Resultant moment vector [Mx,My,Mz] |
| `resultant_force` | float64 | (1, 3) | N | Same as 3D_ResultantForce |
| `resultant_moment` | float64 | (1, 3) | N·m | Same as 3D_ResultantMoment |

## Data Flow and Field Mapping

### 1. Sensor to Robot Data Flow

```
Tac3D Sensor → PyTac3D SDK → Tac3DSensor.read() → ManipulatorRobot → Observation Dict → Dataset
```

### 2. Key Field Mappings

#### Tac3D Sensor Raw Fields (PyTac3D SDK)
```python
# Original fields from PyTac3D SDK
raw_frame = {
    'SN': 'AD2-0046R',                # Sensor serial number
    'index': 314,                     # Frame index
    'recvTimestamp': 22.542121,       # Receiving timestamp
    'sendTimestamp': 12.242941,       # Sending timestamp
    '3D_Positions': np.ndarray,       # (400, 3) positions
    '3D_Displacements': np.ndarray,   # (400, 3) displacements
    '3D_Forces': np.ndarray,          # (400, 3) forces
    '3D_ResultantForce': np.ndarray,  # (1, 3) resultant force
    '3D_ResultantMoment': np.ndarray, # (1, 3) resultant moment
}
```

#### Tac3DSensor.read() Return Fields
```python
# Fields returned by Tac3DSensor.read()
standardized_data = {
    'recvTimestamp': raw_frame.get('recvTimestamp', time.time()),
    'SN': raw_frame.get('SN', 'unknown'),
    'index': raw_frame.get('index', 0),
    '3D_Positions': raw_frame.get('3D_Positions'),
    '3D_Displacements': raw_frame.get('3D_Displacements'), 
    '3D_Forces': raw_frame.get('3D_Forces'),
    '3D_ResultantForce': raw_frame.get('3D_ResultantForce'),
    '3D_ResultantMoment': raw_frame.get('3D_ResultantMoment'),
    # Added standardized field names for compatibility
    'resultant_force': raw_frame.get('3D_ResultantForce'),
    'resultant_moment': raw_frame.get('3D_ResultantMoment'),
    'raw_frame': raw_frame
}
```

#### ManipulatorRobot Expected Fields
```python
# Fields expected by ManipulatorRobot._get_tactile_observation
expected_fields = {
    'SN': 'string value',                 # Sensor serial number
    'index': 0,                           # Frame index (int)
    'recvTimestamp': 22.542121,           # Receiving timestamp (float)
    'sendTimestamp': 12.242941,           # Sending timestamp (float)
    '3D_Positions': np.ndarray(400, 3),   # 3D positions array
    '3D_Displacements': np.ndarray(400, 3), # 3D displacements array
    '3D_Forces': np.ndarray(400, 3),      # 3D forces array
    'resultant_force': np.ndarray(1, 3),  # Resultant force vector
    'resultant_moment': np.ndarray(1, 3), # Resultant moment vector
}
```

#### Dataset Storage Keys
```python
# Keys used in the LeRobot dataset
dataset_keys = {
    # Metadata
    f"observation.tactile.tac3d.{sensor_name}.sensor_sn": "AD2-0046R",
    f"observation.tactile.tac3d.{sensor_name}.frame_index": torch.tensor([314]),
    f"observation.tactile.tac3d.{sensor_name}.send_timestamp": torch.tensor([12.242941]),
    f"observation.tactile.tac3d.{sensor_name}.recv_timestamp": torch.tensor([22.542121]),
    
    # 3D data arrays (converted to PyTorch tensors)
    f"observation.tactile.tac3d.{sensor_name}.positions_3d": torch.tensor(shape=(400, 3)),
    f"observation.tactile.tac3d.{sensor_name}.displacements_3d": torch.tensor(shape=(400, 3)),
    f"observation.tactile.tac3d.{sensor_name}.forces_3d": torch.tensor(shape=(400, 3)),
    
    # Resultant data (flattened to 1D tensors)
    f"observation.tactile.tac3d.{sensor_name}.resultant_force": torch.tensor(shape=(3,)),
    f"observation.tactile.tac3d.{sensor_name}.resultant_moment": torch.tensor(shape=(3,))
}
```

#### Visualization Field Mapping
```python
# Field mapping in visualize_dataset_simple.py
field_mapping = {
    'positions_3d': '3D_Positions',
    'displacements_3d': '3D_Displacements',
    'forces_3d': '3D_Forces',
    'resultant_force': 'resultant_force',  # Use standardized field name
    'resultant_moment': 'resultant_moment',  # Use standardized field name
    'sensor_sn': 'SN',
    'frame_index': 'index',
    'send_timestamp': 'sendTimestamp',
    'recv_timestamp': 'recvTimestamp'
}
```

## Common Issues and Solutions

### 1. Zero Resultant Force

**Problem**: `resultant_force` and `resultant_moment` values are always zero

**Possible causes**:
- Sensor not in contact with any object
- Sensor needs calibration
- Field name mismatch between components

**Solutions**:
1. Check physical contact with the sensor
2. Run `sensor.calibrate()` to reset zero point
3. Verify field names match between components:
   - `resultant_force` vs `3D_ResultantForce`
   - `resultant_moment` vs `3D_ResultantMoment`

### 2. Field Name Mismatches

**Problem**: Data not properly transferred between components

**Key field mappings to check**:
- Sensor SN: `SN` → `sensor_sn`
- Frame index: `index` → `frame_index`
- Timestamps: `recvTimestamp` → `recv_timestamp`, `sendTimestamp` → `send_timestamp`
- Force data: `3D_ResultantForce` → `resultant_force`
- Moment data: `3D_ResultantMoment` → `resultant_moment`

**Solution**: Ensure consistent field names across all components:
1. `tac3d.py`: Return both original and standardized field names
2. `manipulator.py`: Check for both field name variants
3. `visualize_dataset_simple.py`: Use correct field mapping

## Usage Examples

### Reading Sensor Data
```python
from lerobot.common.robot_devices.tactile_sensors.configs import Tac3DConfig
from lerobot.common.robot_devices.tactile_sensors.tac3d import Tac3DSensor

# Create sensor config
config = Tac3DConfig(port=9988, auto_calibrate=True)
sensor = Tac3DSensor(config)
sensor.connect()

# Read data
data = sensor.read()
print(f"Sensor SN: {data['SN']}")
print(f"Resultant force: {data['resultant_force']}")
```

### Accessing Dataset Tactile Data
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset
dataset = LeRobotDataset("user/tactile_dataset")
frame = dataset[0]

# Access tactile data
sensor_name = "main_gripper1"
resultant_force = frame[f"observation.tactile.tac3d.{sensor_name}.resultant_force"]
positions = frame[f"observation.tactile.tac3d.{sensor_name}.positions_3d"]

print(f"Resultant force: {resultant_force}")  # torch.Tensor shape=(3,)
print(f"3D positions shape: {positions.shape}")  # torch.Size([400, 3])
``` 