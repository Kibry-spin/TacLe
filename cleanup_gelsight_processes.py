#!/usr/bin/env python3
"""
清理LeRobot触觉传感器进程脚本

该脚本用于强制清理可能遗留的GelSight和Tac3D传感器进程，
特别是在程序意外终止或无法正常退出时使用。

使用方法:
    python cleanup_gelsight_processes.py
    或
    ./cleanup_gelsight_processes.py
"""

import subprocess
import sys
import os
import signal
import time

def find_and_kill_processes(pattern, description):
    """查找并杀死匹配模式的进程"""
    print(f"正在查找{description}进程...")
    try:
        # 使用pgrep查找进程
        result = subprocess.run(['pgrep', '-f', pattern], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            killed_count = 0
            for pid in pids:
                if pid.strip():
                    try:
                        pid_int = int(pid.strip())
                        # 首先尝试SIGTERM
                        os.kill(pid_int, signal.SIGTERM)
                        print(f"  发送SIGTERM信号到进程 {pid_int}")
                        killed_count += 1
                    except ProcessLookupError:
                        print(f"  进程 {pid_int} 已不存在")
                    except PermissionError:
                        print(f"  没有权限终止进程 {pid_int}")
                    except Exception as e:
                        print(f"  终止进程 {pid_int} 时出错: {e}")
            
            if killed_count > 0:
                print(f"  已发送终止信号到 {killed_count} 个{description}进程")
                # 等待一秒钟让进程正常退出
                time.sleep(1)
                
                # 检查是否还有进程需要强制杀死
                result2 = subprocess.run(['pgrep', '-f', pattern], 
                                       capture_output=True, text=True)
                if result2.returncode == 0:
                    remaining_pids = result2.stdout.strip().split('\n')
                    for pid in remaining_pids:
                        if pid.strip():
                            try:
                                pid_int = int(pid.strip())
                                os.kill(pid_int, signal.SIGKILL)
                                print(f"  强制杀死进程 {pid_int}")
                            except ProcessLookupError:
                                pass
                            except Exception as e:
                                print(f"  强制杀死进程 {pid_int} 时出错: {e}")
            else:
                print(f"  没有找到运行中的{description}进程")
        else:
            print(f"  没有找到{description}进程")
    except FileNotFoundError:
        print("  错误: 找不到pgrep命令")
    except Exception as e:
        print(f"  查找{description}进程时出错: {e}")

def check_ports():
    """检查触觉传感器使用的端口"""
    print("正在检查传感器端口占用情况...")
    try:
        # 检查Tac3D默认端口9988
        result = subprocess.run(['netstat', '-ln'], capture_output=True, text=True)
        if result.returncode == 0:
            if ':9988 ' in result.stdout:
                print("  检测到端口9988被占用（Tac3D传感器）")
            else:
                print("  端口9988未被占用")
    except Exception as e:
        print(f"  检查端口时出错: {e}")

def main():
    print("LeRobot触觉传感器进程清理工具")
    print("=" * 50)
    
    # 清理各种传感器相关进程
    process_patterns = [
        ('ffmpeg.*video', 'FFmpeg视频流（GelSight）'),
        ('python.*gelsight', 'GelSight Python'),
        ('python.*tac3d', 'Tac3D Python'),
        ('python.*tactile', '触觉传感器Python'),
        ('control_robot', 'LeRobot控制脚本'),
    ]
    
    for pattern, description in process_patterns:
        find_and_kill_processes(pattern, description)
        print()
    
    # 检查端口占用
    check_ports()
    
    print("清理完成！")
    print("\n注意:")
    print("- 如果还有进程无法清理，请检查是否有权限问题")
    print("- 可以使用 'ps aux | grep gelsight' 手动检查剩余进程")
    print("- 可以使用 'sudo netstat -tulpn | grep :9988' 检查Tac3D端口")

if __name__ == "__main__":
    main() 