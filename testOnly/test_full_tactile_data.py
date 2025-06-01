#!/usr/bin/env python3
"""
测试完整触觉数据保存功能
"""

import time
import numpy as np
from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.tactile_sensors.configs import Tac3DConfig


def test_full_tactile_data_collection():
    """测试完整触觉数据的收集功能"""
    
    print("=== 测试完整触觉数据收集 ===\n")
    
    # 创建机器人配置
    config = AlohaRobotConfig(
        mock=True,  # 机器人其他部分使用mock
        tactile_sensors={
            "left_gripper": Tac3DConfig(
                port=9988,
                auto_calibrate=True,
                mock=False,  # 使用真实传感器
            ),
        }
    )
    
    print("📋 配置的触觉传感器特征:")
    robot = ManipulatorRobot(config)
    tactile_features = robot.tactile_features
    
    for name, feature in tactile_features.items():
        print(f"  {name}: shape={feature['shape']}, dtype={feature['dtype']}")
    
    print(f"\n总共 {len(tactile_features)} 个触觉数据特征")
    
    try:
        # 连接机器人
        print("\n🔌 正在连接机器人和触觉传感器...")
        robot.connect()
        print("✅ 连接成功!")
        
        # 测试数据收集
        print("\n📊 测试完整触觉数据收集...")
        
        for i in range(3):
            print(f"\n--- 数据采集 #{i+1} ---")
            
            start_time = time.time()
            obs = robot.capture_observation()
            collection_time = time.time() - start_time
            
            print(f"数据收集耗时: {collection_time*1000:.1f}ms")
            
            # 检查每个触觉数据字段
            sensor_name = "left_gripper"
            print(f"\n{sensor_name} 触觉数据详情:")
            
            # 基本信息
            sn_key = f"observation.tactile.{sensor_name}.sensor_sn"
            idx_key = f"observation.tactile.{sensor_name}.frame_index"
            send_ts_key = f"observation.tactile.{sensor_name}.send_timestamp"
            recv_ts_key = f"observation.tactile.{sensor_name}.recv_timestamp"
            
            if sn_key in obs:
                print(f"  传感器SN: {obs[sn_key]}")
                print(f"  帧索引: {obs[idx_key].item()}")
                print(f"  发送时间戳: {obs[send_ts_key].item():.6f}s")
                print(f"  接收时间戳: {obs[recv_ts_key].item():.6f}s")
            
            # 三维数据阵列
            pos_key = f"observation.tactile.{sensor_name}.positions_3d"
            disp_key = f"observation.tactile.{sensor_name}.displacements_3d"
            forces_key = f"observation.tactile.{sensor_name}.forces_3d"
            
            if pos_key in obs:
                positions = obs[pos_key]
                displacements = obs[disp_key]
                forces_3d = obs[forces_key]
                
                print(f"\n  3D位置数据: shape={positions.shape}, dtype={positions.dtype}")
                print(f"    均值: [{positions.mean(dim=0)[0]:.3f}, {positions.mean(dim=0)[1]:.3f}, {positions.mean(dim=0)[2]:.3f}]")
                print(f"    范围: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
                
                print(f"\n  3D位移数据: shape={displacements.shape}, dtype={displacements.dtype}")
                print(f"    位移幅度: {torch.norm(displacements, dim=1).mean():.6f}")
                print(f"    最大位移: {torch.norm(displacements, dim=1).max():.6f}")
                
                print(f"\n  3D力场数据: shape={forces_3d.shape}, dtype={forces_3d.dtype}")
                print(f"    力场幅度: {torch.norm(forces_3d, dim=1).mean():.6f}")
                print(f"    最大力: {torch.norm(forces_3d, dim=1).max():.6f}")
            
            # 合成力和力矩
            force_key = f"observation.tactile.{sensor_name}.resultant_force"
            moment_key = f"observation.tactile.{sensor_name}.resultant_moment"
            
            if force_key in obs:
                force = obs[force_key]
                moment = obs[moment_key]
                
                force_mag = torch.norm(force)
                moment_mag = torch.norm(moment)
                
                print(f"\n  合成力: [{force[0]:.3f}, {force[1]:.3f}, {force[2]:.3f}] |F|={force_mag:.3f}")
                print(f"  合成力矩: [{moment[0]:.3f}, {moment[1]:.3f}, {moment[2]:.3f}] |M|={moment_mag:.3f}")
            
            time.sleep(1)
        
        print("\n📊 数据大小分析:")
        total_size = 0
        for name, feature in tactile_features.items():
            if "shape" in feature:
                shape = feature["shape"]
                if feature["dtype"] == "string":
                    size_bytes = 50  # 估计字符串大小
                elif feature["dtype"] == "int64":
                    size_bytes = np.prod(shape) * 8
                elif feature["dtype"] == "float64":
                    size_bytes = np.prod(shape) * 8
                else:
                    size_bytes = 0
                
                total_size += size_bytes
                print(f"  {name}: {size_bytes} bytes")
        
        print(f"\n每帧触觉数据总大小: {total_size} bytes ({total_size/1024:.1f} KB)")
        print(f"相比之前只保存力和力矩的24字节，现在保存 {total_size/24:.1f}x 的数据量")
        
        print("\n✅ 完整触觉数据收集测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if robot.is_connected:
            print("\n🔌 断开机器人连接...")
            robot.disconnect()
            print("✅ 断开完成")


def test_data_saving_compatibility():
    """测试数据保存兼容性"""
    print("\n=== 测试数据保存兼容性 ===")
    
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import tempfile
    import shutil
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建机器人
        config = AlohaRobotConfig(
            mock=True,
            tactile_sensors={
                "left_gripper": Tac3DConfig(port=9988, mock=False),
            }
        )
        robot = ManipulatorRobot(config)
        robot.connect()
        
        # 创建数据集
        features = robot.features
        print(f"数据集特征数量: {len(features)}")
        
        dataset = LeRobotDataset.create(
            "test_tactile_dataset",
            fps=30,
            root=temp_dir,
            features=features,
            use_videos=False,
        )
        
        print("📝 模拟数据收集和保存...")
        
        # 收集几帧数据
        for i in range(3):
            obs, action = robot.teleop_step(record_data=True)
            frame = {**obs, **action, "task": "test_task"}
            dataset.add_frame(frame)
            print(f"添加第 {i+1} 帧数据")
        
        # 保存episode
        dataset.save_episode()
        print("✅ Episode保存成功!")
        
        # 验证数据
        print(f"数据集长度: {len(dataset)}")
        sample = dataset[0]
        print(f"样本键数量: {len(sample.keys())}")
        
        # 检查触觉数据键
        tactile_keys = [k for k in sample.keys() if "tactile" in k]
        print(f"触觉数据键数量: {len(tactile_keys)}")
        for key in tactile_keys[:5]:  # 显示前几个
            print(f"  {key}: {sample[key].shape if hasattr(sample[key], 'shape') else type(sample[key])}")
        
        robot.disconnect()
        
    except Exception as e:
        print(f"❌ 数据保存测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import torch
    
    # 测试完整数据收集
    test_full_tactile_data_collection()
    
    print("\n" + "="*60 + "\n")
    
    # 测试数据保存兼容性
    test_data_saving_compatibility()
    
    print("\n🎉 所有测试完成!") 