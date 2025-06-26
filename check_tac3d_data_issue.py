#!/usr/bin/env python3
"""
检查Tac3D录制数据问题分析脚本

发现的问题:
1. positions_3d全为0 - 表示传感器未校准或初始化未完成
2. forces_3d全为0 - 但resultant_force有数据，说明数据不一致
3. 传感器元数据缺失 - 序列号、时间戳等都为空或0

问题原因分析:
1. 传感器可能在录制期间未完全初始化
2. Tac3D Desktop软件可能未正确配置
3. 传感器校准可能不正确
4. 网络传输配置可能有问题
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_tac3d_data(dataset_path: str):
    """分析Tac3D数据集的问题"""
    
    dataset_path_obj = Path(dataset_path)
    parquet_file = dataset_path_obj / "data" / "chunk-000" / "episode_000000.parquet"
    
    if not parquet_file.exists():
        print(f"❌ 数据文件不存在: {parquet_file}")
        return
    
    print("🔍 分析Tac3D数据集...")
    print(f"📁 数据路径: {dataset_path}")
    print(f"📄 数据文件: {parquet_file}")
    print()
    
    # 读取数据
    df = pd.read_parquet(parquet_file)
    print(f"📊 数据集信息:")
    print(f"  总帧数: {len(df)}")
    print(f"  总列数: {len(df.columns)}")
    
    # 找出Tac3D相关列
    tac3d_cols = [col for col in df.columns if 'tac3d' in col]
    print(f"  Tac3D相关列数: {len(tac3d_cols)}")
    print()
    
    # 检查各种数据的问题
    problems = []
    
    # 1. 检查传感器元数据
    print("🔍 传感器元数据检查:")
    
    sn_col = 'observation.tactile.tac3d.main_gripper1.sensor_sn'
    if sn_col in df.columns:
        sn_values = df[sn_col].unique()
        if len(sn_values) == 1 and (sn_values[0] == '' or pd.isna(sn_values[0])):
            problems.append("传感器序列号为空")
            print("  ❌ 传感器序列号: 空值")
        else:
            print(f"  ✅ 传感器序列号: {sn_values}")
    
    # 检查帧索引
    fi_col = 'observation.tactile.tac3d.main_gripper1.frame_index'
    if fi_col in df.columns:
                 fi_values = df[fi_col].unique()
         if len(fi_values) == 1 and fi_values[0] == 0:
             problems.append("传感器帧索引始终为0")
             print("  ❌ 传感器帧索引: 始终为0")
         else:
             print(f"  ✅ 传感器帧索引: 范围 {min(fi_values)}-{max(fi_values)}")
    
    # 检查时间戳
    st_col = 'observation.tactile.tac3d.main_gripper1.send_timestamp'
    rt_col = 'observation.tactile.tac3d.main_gripper1.recv_timestamp'
    if st_col in df.columns:
        st_values = df[st_col].unique()
        if len(st_values) == 1 and st_values[0] == 0.0:
            problems.append("发送时间戳始终为0")
            print("  ❌ 发送时间戳: 始终为0")
        else:
            print(f"  ✅ 发送时间戳: 范围 {st_values.min():.3f}-{st_values.max():.3f}s")
    
    print()
    
    # 2. 检查3D数据矩阵
    print("🔍 3D数据矩阵检查:")
    
    # 检查positions_3d
    pos_col = 'observation.tactile.tac3d.main_gripper1.positions_3d'
    if pos_col in df.columns:
        pos_sample = df[pos_col].iloc[0]
        pos_matrix = np.array([pos_sample[i] for i in range(len(pos_sample))])
        
        if np.all(pos_matrix == 0):
            problems.append("3D位置数据全为0")
            print("  ❌ positions_3d: 全为0 (传感器未校准)")
        else:
            print(f"  ✅ positions_3d: 有效数据，范围 {pos_matrix.min():.3f}-{pos_matrix.max():.3f}")
    
    # 检查forces_3d
    force_col = 'observation.tactile.tac3d.main_gripper1.forces_3d'
    if force_col in df.columns:
        force_sample = df[force_col].iloc[0]
        force_matrix = np.array([force_sample[i] for i in range(len(force_sample))])
        
        if np.all(force_matrix == 0):
            problems.append("3D力场数据全为0")
            print("  ❌ forces_3d: 全为0 (但resultant_force有数据)")
        else:
            print(f"  ✅ forces_3d: 有效数据，非零元素 {np.count_nonzero(force_matrix)}")
    
    # 检查resultant_force
    rf_col = 'observation.tactile.tac3d.main_gripper1.resultant_force'
    if rf_col in df.columns:
        non_zero_count = 0
        max_magnitude = 0
        for i in range(len(df)):
            rf = df[rf_col].iloc[i]
            magnitude = np.linalg.norm(rf)
            if magnitude > 0.01:
                non_zero_count += 1
                max_magnitude = max(max_magnitude, magnitude)
        
        if non_zero_count > 0:
            print(f"  ✅ resultant_force: {non_zero_count}/{len(df)}帧有数据，最大幅度 {max_magnitude:.3f}N")
        else:
            problems.append("合力数据全为0")
            print("  ❌ resultant_force: 全为0")
    
    print()
    
    # 3. 数据一致性检查
    print("🔍 数据一致性检查:")
    
    if force_col in df.columns and rf_col in df.columns:
        # 找一个有resultant_force的帧
        test_frame = None
        for i in range(len(df)):
            rf = df[rf_col].iloc[i]
            if np.linalg.norm(rf) > 0.1:
                test_frame = i
                break
        
        if test_frame is not None:
            rf = df[rf_col].iloc[test_frame]
            force_data = df[force_col].iloc[test_frame]
            force_matrix = np.array([force_data[i] for i in range(len(force_data))])
            force_sum = np.sum(force_matrix, axis=0)
            
            diff = np.linalg.norm(rf - force_sum)
            if diff > 0.001:
                problems.append("forces_3d与resultant_force不一致")
                print(f"  ❌ 数据一致性: forces_3d总和与resultant_force差异 {diff:.6f}")
                print(f"      resultant_force: {rf}")
                print(f"      forces_3d总和: {force_sum}")
            else:
                print("  ✅ 数据一致性: forces_3d与resultant_force匹配")
        else:
            print("  ⚠️  无法检查一致性: 没有有效的force数据")
    
    print()
    
    # 4. 问题总结
    print("📋 问题总结:")
    if problems:
        print("  发现以下问题:")
        for i, problem in enumerate(problems, 1):
            print(f"    {i}. {problem}")
    else:
        print("  ✅ 未发现明显问题")
    
    print()
    
    # 5. 解决方案建议
    print("💡 解决方案建议:")
    
    if "传感器序列号为空" in problems or "传感器帧索引始终为0" in problems:
        print("  1. 🔧 传感器连接问题:")
        print("     - 检查Tac3D Desktop是否正常运行")
        print("     - 确认传感器已正确连接并被识别")
        print("     - 验证UDP端口9988未被占用")
    
    if "3D位置数据全为0" in problems:
        print("  2. 📐 传感器校准问题:")
        print("     - 在Tac3D Desktop中执行传感器校准")
        print("     - 确保传感器初始化完成（等待100帧）")
        print("     - 检查传感器配置文件")
    
    if "3D力场数据全为0" in problems and "合力数据全为0" not in problems:
        print("  3. ⚙️  数据配置问题:")
        print("     - 检查Tac3D Desktop的数据输出配置")
        print("     - 确认forces_3d字段已启用传输")
        print("     - 重新启动Tac3D Desktop并重新连接")
    
    if "forces_3d与resultant_force不一致" in problems:
        print("  4. 🔄 数据处理问题:")
        print("     - 可能是数据处理流程中的bug")
        print("     - 建议重新录制数据")
        print("     - 联系技术支持检查数据处理逻辑")
    
    print("\n🚀 推荐操作顺序:")
    print("  1. 重启Tac3D Desktop软件")
    print("  2. 重新连接并校准传感器")
    print("  3. 等待传感器完全初始化（观察帧索引递增）")
    print("  4. 确认3D数据正常显示后再开始录制")
    print("  5. 录制时确保有物理接触产生force数据")

def main():
    if len(sys.argv) != 2:
        print("用法: python check_tac3d_data_issue.py <数据集路径>")
        print("示例: python check_tac3d_data_issue.py /home/user/.cache/huggingface/lerobot/user/test_two2")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    analyze_tac3d_data(dataset_path)

if __name__ == "__main__":
    main() 