#!/usr/bin/env python3
"""
清理 GelSight 和 TAC3D 传感器相关的进程和端口占用

Usage:
    python cleanup_gelsight_processes.py
    python cleanup_gelsight_processes.py --port 9988
    python cleanup_gelsight_processes.py --all
"""

import argparse
import subprocess
import sys
import time
import signal
import os

def get_processes_using_ports(ports):
    """获取占用指定端口的进程列表"""
    processes = []
    try:
        # 使用 netstat 查找占用端口的进程
        result = subprocess.run(['netstat', '-tulpn'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            for port in ports:
                if f":{port}" in line:
                    parts = line.split()
                    for part in parts:
                        if '/' in part:
                            try:
                                pid = int(part.split('/')[0])
                                process_name = part.split('/')[1]
                                processes.append({
                                    'pid': pid,
                                    'name': process_name,
                                    'port': port,
                                    'line': line.strip()
                                })
                            except (ValueError, IndexError):
                                continue
    except Exception as e:
        print(f"Error getting processes: {e}")
    
    return processes

def get_gelsight_processes():
    """获取 GelSight 相关的进程"""
    processes = []
    try:
        # 查找包含 gelsight 关键词的进程
        result = subprocess.run(['pgrep', '-f', 'gelsight'], capture_output=True, text=True)
        pids = [int(pid.strip()) for pid in result.stdout.split() if pid.strip()]
        
        for pid in pids:
            try:
                # 获取进程详细信息
                cmd_result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,ppid,cmd'], 
                                          capture_output=True, text=True)
                if cmd_result.returncode == 0:
                    lines = cmd_result.stdout.strip().split('\n')
                    if len(lines) > 1:  # 跳过标题行
                        process_info = lines[1].strip().split(None, 2)
                        if len(process_info) >= 3:
                            processes.append({
                                'pid': pid,
                                'ppid': process_info[1],
                                'cmd': process_info[2],
                                'type': 'gelsight'
                            })
            except Exception:
                continue
                
    except Exception as e:
        print(f"Error getting GelSight processes: {e}")
    
    return processes

def cleanup_port(port):
    """清理指定端口的进程"""
    print(f"\n🔍 检查端口 {port}...")
    
    processes = get_processes_using_ports([port])
    if not processes:
        print(f"✅ 端口 {port} 未被占用")
        return True
    
    print(f"❌ 发现 {len(processes)} 个进程占用端口 {port}:")
    for proc in processes:
        print(f"  - PID: {proc['pid']}, 进程: {proc['name']}")
        print(f"    详情: {proc['line']}")
    
    # 尝试杀死进程
    success = True
    for proc in processes:
        try:
            pid = proc['pid']
            print(f"\n🔪 正在终止进程 {pid} ({proc['name']})...")
            
            # 首先尝试优雅关闭 (SIGTERM)
            os.kill(pid, signal.SIGTERM)
            time.sleep(1.0)
            
            # 检查进程是否还存在
            try:
                os.kill(pid, 0)  # 不发送信号，只检查进程是否存在
                print(f"   进程 {pid} 仍在运行，使用强制终止 (SIGKILL)...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
            except ProcessLookupError:
                print(f"   ✅ 进程 {pid} 已终止")
            
        except ProcessLookupError:
            print(f"   ✅ 进程 {pid} 已不存在")
        except PermissionError:
            print(f"   ❌ 权限不足，无法终止进程 {pid}")
            success = False
        except Exception as e:
            print(f"   ❌ 终止进程 {pid} 失败: {e}")
            success = False
    
    # 再次检查端口
    time.sleep(1.0)
    remaining = get_processes_using_ports([port])
    if remaining:
        print(f"❌ 端口 {port} 仍被占用:")
        for proc in remaining:
            print(f"  - PID: {proc['pid']}, 进程: {proc['name']}")
        return False
    else:
        print(f"✅ 端口 {port} 已释放")
        return True

def cleanup_gelsight_processes():
    """清理 GelSight 相关进程"""
    print("\n🔍 查找 GelSight 相关进程...")
    
    processes = get_gelsight_processes()
    if not processes:
        print("✅ 未发现 GelSight 相关进程")
        return True
    
    print(f"❌ 发现 {len(processes)} 个 GelSight 相关进程:")
    for proc in processes:
        print(f"  - PID: {proc['pid']}, PPID: {proc['ppid']}")
        print(f"    命令: {proc['cmd']}")
    
    # 终止进程
    success = True
    for proc in processes:
        try:
            pid = proc['pid']
            print(f"\n🔪 正在终止 GelSight 进程 {pid}...")
            
            # 优雅关闭
            os.kill(pid, signal.SIGTERM)
            time.sleep(1.0)
            
            # 检查是否还存在
            try:
                os.kill(pid, 0)
                print(f"   进程 {pid} 仍在运行，强制终止...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
            except ProcessLookupError:
                print(f"   ✅ 进程 {pid} 已终止")
            
        except ProcessLookupError:
            print(f"   ✅ 进程 {pid} 已不存在")
        except PermissionError:
            print(f"   ❌ 权限不足，无法终止进程 {pid}")
            success = False
        except Exception as e:
            print(f"   ❌ 终止进程 {pid} 失败: {e}")
            success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(
        description="清理 GelSight 和 TAC3D 传感器相关的进程和端口占用"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        help="清理指定端口 (默认: 9988 for TAC3D)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="清理所有相关进程和端口"
    )
    parser.add_argument(
        "--gelsight-only", 
        action="store_true", 
        help="只清理 GelSight 相关进程"
    )
    
    args = parser.parse_args()
    
    print("🧹 传感器进程清理工具")
    print("=" * 50)
    
    success = True
    
    if args.all:
        # 清理所有相关进程和端口
        print("🔄 执行完整清理...")
        
        # 清理 GelSight 进程
        if not cleanup_gelsight_processes():
            success = False
        
        # 清理常用的 TAC3D 端口
        tac3d_ports = [9988, 9989, 9990, 9991, 9992]
        for port in tac3d_ports:
            if not cleanup_port(port):
                success = False
                
    elif args.gelsight_only:
        # 只清理 GelSight 进程
        if not cleanup_gelsight_processes():
            success = False
            
    elif args.port:
        # 清理指定端口
        if not cleanup_port(args.port):
            success = False
            
    else:
        # 默认：清理 TAC3D 默认端口
        print("🔄 清理 TAC3D 默认端口 9988...")
        if not cleanup_port(9988):
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 清理完成！")
        sys.exit(0)
    else:
        print("❌ 清理过程中遇到问题，请检查日志")
        sys.exit(1)

if __name__ == "__main__":
    main() 