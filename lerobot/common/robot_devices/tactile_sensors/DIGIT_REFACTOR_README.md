# DIGIT传感器重构说明

## 概述

已将DIGIT传感器实现重构为使用`digit-interface`库，而不是直接使用`gs_sdk`。

## 主要更改

### 1. digit.py文件重构

**之前**: 直接使用`gs_sdk.gs_device.Camera`
**现在**: 使用`digit_interface.digit.Digit`和`digit_interface.digit_handler.DigitHandler`

#### 关键变化:
- 导入`digit_interface`库而不是`gs_sdk`
- 使用`Digit`类而不是`Camera`类
- 自动配置分辨率和帧率（QVGA, 60fps或30fps）
- 自动设置LED强度为最大值
- 图像格式从BGR转换为RGB
- 保持与原有接口的兼容性

### 2. demo_digit.py修复

修复了linter错误：
```python
# 之前 (有bug):
digit = DigitHandler.find_digit("D12345")
cap = cv2.VideoCapture(digit["dev_name"])  # 如果digit为None会报错

# 现在 (已修复):
digit = DigitHandler.find_digit("D12345")
if digit is not None:
    cap = cv2.VideoCapture(digit["dev_name"])
    cap.release()
else:
    print("DIGIT device D12345 not found")
```

### 3. manipulator.py兼容性

- **无需更改**: `manipulator.py`已经包含对DIGIT传感器的完整支持
- 支持图像数据和力数据（即使力数据为零值）
- 在`tactile_features`属性中定义了DIGIT传感器的数据结构
- 在`_get_tactile_observation`方法中处理DIGIT传感器数据

## 使用方法

### 基本配置

```python
from lerobot.common.robot_devices.tactile_sensors.configs import DIGITConfig
from lerobot.common.robot_devices.tactile_sensors.digit import DIGITSensor

# 配置DIGIT传感器（使用实际序列号）
config = DIGITConfig(
    device_name="D21186",  # 你的DIGIT序列号
    imgh=240,
    imgw=320,
    framerate=60
)

# 创建传感器实例
sensor = DIGITSensor(config)
sensor.connect()

# 读取数据
data = sensor.read()
image = data['tactile_image']  # numpy array (240, 320, 3)

sensor.disconnect()
```

### 在机器人配置中使用

```python
from lerobot.common.robot_devices.robots.configs import So101RobotConfig

config = So101RobotConfig()
config.tactile_sensors = {
    "gripper_digit": DIGITConfig(
        device_name="D21186",  # 你的DIGIT序列号
        imgh=240,
        imgw=320,
        framerate=60,
    ),
}

# 使用机器人
robot = ManipulatorRobot(config)
robot.connect()
observation = robot.capture_observation()
```

## 数据格式

DIGIT传感器返回的数据结构：

```python
{
    'tactile_image': np.ndarray,      # (240, 320, 3) RGB图像
    'timestamp': float,               # 时间戳
    'frame_index': int,               # 帧索引
    'device_name': str,               # 设备名称
    'resultant_force': np.ndarray,    # (3,) 力向量（通常为零）
    'resultant_moment': np.ndarray,   # (3,) 力矩向量（通常为零）
}
```

## 设备发现

可以使用`DigitHandler`查找连接的DIGIT设备：

```python
from digit_interface.digit_handler import DigitHandler

# 列出所有DIGIT设备
digits = DigitHandler.list_digits()
for digit in digits:
    print(f"Serial: {digit['serial']}, Device: {digit['dev_name']}")

# 根据序列号查找特定设备
digit = DigitHandler.find_digit("D21186")
if digit:
    print(f"Found: {digit}")
```

## 注意事项

1. **序列号**: 必须使用实际的DIGIT设备序列号（如"D21186"）
2. **权限**: 确保当前用户有访问USB设备的权限
3. **依赖**: 需要安装`digit-interface`库
4. **兼容性**: 新实现完全兼容现有的`manipulator.py`接口

## 测试

可以运行以下测试确认DIGIT传感器工作正常：

```bash
# 测试基本功能
python lerobot/common/robot_devices/tactile_sensors/digit.py

# 测试demo
python lerobot/common/robot_devices/tactile_sensors/digit-interface/example/demo_digit.py
``` 