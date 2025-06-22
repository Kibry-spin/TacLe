# GelSight 触觉传感器集成

本文档介绍如何在LeRobot中使用GelSight触觉传感器。

## 安装

### 1. 安装 gs_sdk
```bash
# 进入 gs_sdk 目录
cd lerobot/common/robot_devices/tactile_sensors/gs_sdk

# 安装（开发模式，推荐）
pip install -e .
```

### 2. 验证安装
```bash
# 运行集成测试
cd lerobot/common/robot_devices/tactile_sensors
python test_gelsight.py
```

## 基本用法

### 1. 配置传感器
```python
from lerobot.common.robot_devices.tactile_sensors.configs import GelSightConfig

# 默认配置
config = GelSightConfig()

# 自定义配置
config = GelSightConfig(
    device_name="GelSight Mini",
    imgh=240,
    imgw=320,
    framerate=30
)
```

### 2. 使用传感器
```python
from lerobot.common.robot_devices.tactile_sensors.gelsight import GelSightSensor

# 创建传感器实例
sensor = GelSightSensor(config)

# 连接
sensor.connect()

# 读取数据
data = sensor.read()
image = data['image']  # 触觉图像
timestamp = data['timestamp']  # 时间戳

# 断开连接
sensor.disconnect()
```

## 示例程序

### 1. 基本数据读取
```bash
python examples/gelsight_example.py --mode basic --duration 10
```

### 2. 数据记录
```bash
python examples/gelsight_example.py --mode record --output_dir ./tactile_data --duration 5
```

### 3. 实时可视化
```bash
python examples/gelsight_example.py --mode visualize
```

### 4. 直接测试传感器
```bash
# 测试默认设备
python gelsight.py

# 测试特定设备
python gelsight.py --device "GelSight Mini" --duration 10
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device_name` | str | "GelSight Mini" | 相机设备名称 |
| `imgh` | int | 240 | 期望的图像高度 |
| `imgw` | int | 320 | 期望的图像宽度 |
| `raw_imgh` | int | 2464 | 原始图像高度 |
| `raw_imgw` | int | 3280 | 原始图像宽度 |
| `framerate` | int | 25 | 帧率 |
| `config_path` | str | "" | 配置文件路径（可选） |

## 数据格式

传感器返回的数据字典包含：

```python
{
    'timestamp': float,        # 时间戳
    'device_name': str,        # 设备名称
    'frame_index': int,        # 帧索引
    'image': np.ndarray,       # 触觉图像 (H, W, 3)
    'image_shape': tuple,      # 图像形状
    'sensor_config': dict      # 传感器配置信息
}
```

## 在机器人控制中使用

可以在LeRobot的机器人配置中添加GelSight传感器：

```yaml
# robot_config.yaml
tactile_sensors:
  gelsight_finger:
    type: gelsight
    device_name: "GelSight Mini"
    framerate: 25
```

## 故障排除

### 1. 导入错误
确保已正确安装gs_sdk：
```bash
pip install -e lerobot/common/robot_devices/tactile_sensors/gs_sdk
```

### 2. 设备未找到
检查相机设备是否连接：
```bash
ls /dev/video*
```

### 3. 权限问题
确保用户有访问摄像头的权限：
```bash
sudo usermod -a -G video $USER
```

### 4. 依赖冲突
如果遇到numpy版本冲突，可以更新requirements：
```bash
pip install --upgrade numpy
```

## 高级用法

### 1. 使用配置文件
```python
config = GelSightConfig(config_path="path/to/gsmini.yaml")
sensor = GelSightSensor(config)
```

### 2. 异步读取
```python
# 与同步读取相同，GelSight传感器不需要特殊的异步处理
data = sensor.async_read()
```

### 3. 获取传感器信息
```python
info = sensor.get_sensor_info()
print(f"设备: {info['device_name']}")
print(f"连接状态: {info['is_connected']}")
print(f"帧数: {info['frame_count']}")
```

## 性能优化

1. **帧率设置**: 根据应用需求调整framerate
2. **图像尺寸**: 根据需要调整imgh和imgw
3. **缓冲区**: FastCamera已优化低延迟性能

## 与Tac3D的区别

| 特性 | GelSight | Tac3D |
|------|----------|--------|
| 数据类型 | 图像 | 3D力/位移 |
| 接口 | USB摄像头 | UDP网络 |
| 延迟 | 低（视觉） | 极低（数值） |
| 数据量 | 大（图像） | 小（数值） | 