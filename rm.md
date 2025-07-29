# 🤖 LeRobot with Tactile Sensing

<div align="center">
  <img src="./media/lerobot-logo-light.png" alt="LeRobot Logo" width="400"/>
</div>

## 📝 Project Overview

This project integrates tactile sensing capabilities (GelSight & others) into the LeRobot framework, enhancing the robot's perception abilities through touch.

## ✨ Key Features

- Integration of GelSight tactile sensors for precise touch feedback
- Support for multiple tactile sensor types
- Real-time tactile data processing
- Tactile-based control capabilities
- Tactile data visualization tools

## 🛠️ Installation Guide

### 1. Install Dependencies

```bash
# Install LeRobot
pip install -e .

# Install servo SDK
pip install feetech-servo-sdk

# Install configuration parser
pip install ruamel.yaml
```

### 2. Install GelSight SDK

```bash
# Navigate to tactile sensors directory
cd ./lerobot/common/robot_devices/tactile_sensors

# Clone GelSight SDK repository
git clone https://github.com/joehjhuang/gs_sdk.git

# Important: Modify setup.py to comment out numpy dependency
# Open setup.py and comment out the numpy line as follows:
# "# numpy==1.26.4,"

# Install GelSight SDK
cd gs_sdk
pip install -e .
```

> **Note:** The numpy dependency in gs_sdk's setup.py must be commented out to avoid version conflicts. This is already done in the repository, but if you're using a different version, make sure to check.

## 🚀 Usage

### Run Tactile Robot Control Example

```bash
python -m lerobot.scripts.control_robot
```

### Tactile Data Access Example

```python
import lerobot

# Initialize robot with tactile sensors
robot = lerobot.make_robot(config)

# Access tactile data
tactile_data = robot.get_tactile_data()

# Process tactile feedback
robot.process_tactile_feedback()
```

## 📚 Related Links

- [GelSight SDK](https://github.com/joehjhuang/gs_sdk)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)

## 📄 License

This project is licensed under the Apache 2.0 License.

## 👥 Contributing

Contributions are welcome! Please refer to [Contributing Guidelines](./project_files/CONTRIBUTING.md) for more information.
