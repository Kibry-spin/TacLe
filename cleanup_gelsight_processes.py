#!/usr/bin/env python3
"""
清理GelSight相关的遗留进程
用于解决资源占用问题，避免重启电脑
"""

import subprocess
import re
import os
import signal
import time


def find_gelsight_processes():
    """查找GelSight相关的进程"""
    gelsight_processes = []
    
    try:
        # 查找ffmpeg进程（GelSight使用ffmpeg获取视频流）
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if 'ffmpeg' in line and ('/dev/video' in line or 'GelSight' in line):
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    gelsight_processes.append({
                        'pid': pid,
                        'type': 'ffmpeg',
                        'command': ' '.join(parts[10:])
                    })
        
        # 查找可能的python进程（更精确的匹配）
        for line in lines:
            if 'python' in line and ('gelsight' in line.lower() or 'tactile' in line.lower() or 'cleanup_gelsight' in line.lower()):
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    # 排除当前清理脚本自己
                    current_pid = str(os.getpid())
                    if pid != current_pid:
                        gelsight_processes.append({
                            'pid': pid,
                            'type': 'python',
                            'command': ' '.join(parts[10:])
                        })
    
    except Exception as e:
        print(f"查找进程时出错: {e}")
    
    return gelsight_processes


def check_video_devices():
    """检查视频设备的使用情况"""
    print("=== 检查视频设备状态 ===")
    
    try:
        # 使用lsof检查视频设备
        result = subprocess.run(['lsof', '/dev/video*'], capture_output=True, text=True)
        if result.stdout:
            print("被占用的视频设备:")
            print(result.stdout)
        else:
            print("没有视频设备被占用")
    except subprocess.CalledProcessError:
        print("没有视频设备被占用")
    except Exception as e:
        print(f"检查视频设备时出错: {e}")


def kill_process(pid, process_type):
    """终止指定的进程"""
    try:
        # 首先尝试温和终止
        os.kill(int(pid), signal.SIGTERM)
        print(f"发送SIGTERM到{process_type}进程 {pid}")
        
        # 等待一秒看是否终止
        time.sleep(1)
        
        # 检查进程是否还存在
        try:
            os.kill(int(pid), 0)  # 不发送信号，只检查进程是否存在
            # 如果没有抛出异常，说明进程还存在，强制终止
            print(f"进程 {pid} 仍在运行，强制终止...")
            os.kill(int(pid), signal.SIGKILL)
            print(f"已强制终止{process_type}进程 {pid}")
        except ProcessLookupError:
            # 进程已经不存在了
            print(f"{process_type}进程 {pid} 已成功终止")
            
    except ProcessLookupError:
        print(f"进程 {pid} 不存在")
    except PermissionError:
        print(f"没有权限终止进程 {pid}，可能需要sudo")
    except Exception as e:
        print(f"终止进程 {pid} 时出错: {e}")


def kill_all_matching_processes():
    """持续查找并终止匹配的进程，直到没有新进程"""
    print("=== 持续清理模式 ===")
    
    killed_count = 0
    max_iterations = 10  # 最多尝试10次
    
    for iteration in range(max_iterations):
        print(f"\n第 {iteration + 1} 次扫描...")
        processes = find_gelsight_processes()
        
        if not processes:
            print("没有找到更多进程")
            break
        
        print(f"找到 {len(processes)} 个进程:")
        for proc in processes:
            print(f"  PID: {proc['pid']}, 类型: {proc['type']}")
            kill_process(proc['pid'], proc['type'])
            killed_count += 1
        
        # 等待一下让进程完全终止
        time.sleep(0.5)
    
    print(f"\n总共终止了 {killed_count} 个进程")
    
    # 最终检查
    final_processes = find_gelsight_processes()
    if final_processes:
        print(f"警告: 仍有 {len(final_processes)} 个进程在运行")
        for proc in final_processes:
            print(f"  残留进程 PID: {proc['pid']}, 类型: {proc['type']}")
    else:
        print("所有相关进程已清理完毕")


def force_kill_by_pattern():
    """使用系统命令强制终止所有匹配的进程"""
    print("=== 强制终止模式 ===")
    
    # 1. 终止所有ffmpeg进程（使用/dev/video的）
    try:
        print("查找并终止ffmpeg进程...")
        result = subprocess.run([
            'pkill', '-f', '/dev/video.*ffmpeg|ffmpeg.*dev/video'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("成功终止ffmpeg进程")
        else:
            print("没有找到ffmpeg进程或已终止")
    except Exception as e:
        print(f"终止ffmpeg进程时出错: {e}")
    
    # 2. 终止包含gelsight关键词的python进程
    try:
        print("查找并终止gelsight相关的python进程...")
        result = subprocess.run([
            'pkill', '-f', 'python.*gelsight|python.*tactile'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("成功终止python进程")
        else:
            print("没有找到相关python进程或已终止")
    except Exception as e:
        print(f"终止python进程时出错: {e}")
    
    time.sleep(2)  # 等待进程终止
    
    # 3. 强制终止（SIGKILL）
    try:
        print("强制终止残留进程...")
        subprocess.run(['pkill', '-9', '-f', '/dev/video.*ffmpeg'], capture_output=True)
        subprocess.run(['pkill', '-9', '-f', 'python.*gelsight'], capture_output=True)
        print("强制终止完成")
    except Exception as e:
        print(f"强制终止时出错: {e}")


def release_video_devices():
    """尝试释放视频设备"""
    print("=== 释放视频设备 ===")
    
    # 查找占用视频设备的进程
    try:
        result = subprocess.run(['lsof', '/dev/video*'], capture_output=True, text=True)
        if result.stdout:
            print("占用视频设备的进程:")
            lines = result.stdout.split('\n')[1:]  # 跳过标题行
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        print(f"  PID {pid}: {' '.join(parts[8:])}")
                        # 终止占用设备的进程
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"    已发送SIGTERM到进程 {pid}")
                            time.sleep(0.5)
                            # 检查是否还在运行
                            try:
                                os.kill(int(pid), 0)
                                print(f"    强制终止进程 {pid}")
                                os.kill(int(pid), signal.SIGKILL)
                            except ProcessLookupError:
                                print(f"    进程 {pid} 已终止")
                        except Exception as e:
                            print(f"    终止进程 {pid} 失败: {e}")
        else:
            print("没有进程占用视频设备")
    except subprocess.CalledProcessError:
        print("没有进程占用视频设备")
    except Exception as e:
        print(f"检查视频设备时出错: {e}")


def main():
    print("=== GelSight 进程清理工具 ===")
    
    # 检查视频设备状态
    check_video_devices()
    
    # 查找相关进程
    print("\n=== 查找GelSight相关进程 ===")
    processes = find_gelsight_processes()
    
    if not processes:
        print("没有找到GelSight相关进程")
        return
    
    print("找到以下进程:")
    for i, proc in enumerate(processes):
        print(f"{i+1}. PID: {proc['pid']}, 类型: {proc['type']}")
        print(f"   命令: {proc['command'][:80]}...")
    
    # 询问用户是否要终止这些进程
    print("\n选择操作:")
    print("1. 终止所有找到的进程")
    print("2. 选择性终止进程")
    print("3. 持续清理模式（自动查找新进程）")
    print("4. 强制终止模式（使用系统命令）")
    print("5. 释放视频设备")
    print("6. 仅显示，不终止")
    print("0. 退出")
    
    try:
        choice = input("请输入选择 (0-6): ").strip()
        
        if choice == '0':
            print("退出")
            return
        elif choice == '6':
            print("仅显示模式，不执行终止操作")
            return
        elif choice == '1':
            print("终止所有进程...")
            for proc in processes:
                kill_process(proc['pid'], proc['type'])
        elif choice == '2':
            print("选择要终止的进程 (输入编号，多个用空格分隔):")
            indices_str = input("编号: ").strip()
            try:
                indices = [int(x) - 1 for x in indices_str.split()]
                for idx in indices:
                    if 0 <= idx < len(processes):
                        proc = processes[idx]
                        kill_process(proc['pid'], proc['type'])
                    else:
                        print(f"无效的编号: {idx + 1}")
            except ValueError:
                print("输入格式错误")
        elif choice == '3':
            kill_all_matching_processes()
        elif choice == '4':
            force_kill_by_pattern()
        elif choice == '5':
            release_video_devices()
        else:
            print("无效选择")
    
    except KeyboardInterrupt:
        print("\n被用户中断")
    except Exception as e:
        print(f"操作过程中出错: {e}")
    
    # 再次检查设备状态
    print("\n=== 清理后的设备状态 ===")
    check_video_devices()


def super_cleanup():
    """超级清理模式 - 使用所有可用的清理方法"""
    print("=== 超级清理模式 ===")
    
    print("步骤1: 常规清理...")
    processes = find_gelsight_processes()
    for proc in processes:
        kill_process(proc['pid'], proc['type'])
    
    time.sleep(1)
    
    print("\n步骤2: 持续清理...")
    kill_all_matching_processes()
    
    time.sleep(1)
    
    print("\n步骤3: 强制终止...")
    force_kill_by_pattern()
    
    time.sleep(1)
    
    print("\n步骤4: 释放视频设备...")
    release_video_devices()
    
    print("\n超级清理完成!")


def quick_cleanup():
    """快速清理模式 - 自动终止所有相关进程"""
    print("=== 快速清理模式 ===")
    
    processes = find_gelsight_processes()
    if not processes:
        print("没有找到需要清理的进程")
        return
    
    print(f"找到 {len(processes)} 个相关进程，正在清理...")
    for proc in processes:
        print(f"终止 {proc['type']} 进程 {proc['pid']}")
        kill_process(proc['pid'], proc['type'])
    
    # 追加持续清理
    time.sleep(0.5)
    kill_all_matching_processes()
    
    print("快速清理完成")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            quick_cleanup()
        elif sys.argv[1] == '--super':
            super_cleanup()
        elif sys.argv[1] == '--force':
            force_kill_by_pattern()
        else:
            print("可用参数: --quick, --super, --force")
    else:
        main() 