#!/usr/bin/env python3
import socket
import time

print("创建UDP接收器，监听端口9988...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 9988))
sock.settimeout(1.0)

print("UDP接收器已启动，等待数据...")
start_time = time.time()
packet_count = 0

try:
    while time.time() - start_time < 20:  # 20秒超时
        try:
            data, addr = sock.recvfrom(65535)
            packet_count += 1
            print(f"收到数据包 #{packet_count} 来自 {addr[0]}:{addr[1]}, 大小: {len(data)}字节")
            if packet_count == 1:
                print(f"数据前20字节: {data[:20]}")
        except socket.timeout:
            print(f"等待中... 已等待 {time.time() - start_time:.1f}秒")
            continue
except KeyboardInterrupt:
    print("用户中断")
finally:
    sock.close()
    
print(f"\n总共接收: {packet_count} 个数据包")
if packet_count == 0:
    print("❌ 未接收到数据，请检查TAC3D软件配置")
else:
    print("✅ UDP通信正常")
