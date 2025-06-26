 #!/usr/bin/env python3
import socket
import threading
import time

def udp_sender(port, num_packets=10):
    """
    Sends UDP packets to the specified port on localhost.
    """
    host = '127.0.0.1'
    sent_count = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        print(f"[发送方] 将发送 {num_packets} 个数据包到 {host}:{port}")
        for i in range(num_packets):
            message = f"测试数据包 #{i+1}".encode('utf-8')
            s.sendto(message, (host, port))
            sent_count += 1
            time.sleep(0.2)
    print(f"[发送方] 发送完成，共发送 {sent_count} 个包.")

def udp_receiver(port, received_packets_list):
    """
    Receives UDP packets on the specified port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('0.0.0.0', port))
        s.settimeout(1.0)
        print(f"[接收方] 在端口 {port} 上监听...")
        
        # Listen for a bit longer than the sender takes to send all packets.
        # The sender sends 10 packets with 0.2s sleep, taking 2s. We'll wait for 3s.
        end_time = time.time() + 3
        while time.time() < end_time:
            try:
                data, addr = s.recvfrom(1024)
                print(f"[接收方] 从 {addr} 收到: {data.decode('utf-8')}")
                received_packets_list.append(data)
            except socket.timeout:
                continue
    print(f"[接收方] 监听结束. 共收到 {len(received_packets_list)} 个包.")

if __name__ == "__main__":
    PORT = 9988
    NUM_PACKETS_TO_SEND = 10
    
    received_packets = []
    
    receiver_thread = threading.Thread(target=udp_receiver, args=(PORT, received_packets))
    sender_thread = threading.Thread(target=udp_sender, args=(PORT, NUM_PACKETS_TO_SEND))
    
    print(f"开始在端口 {PORT} 进行UDP收发测试...")
    
    receiver_thread.start()
    time.sleep(0.1)  # Give receiver a moment to bind the socket
    sender_thread.start()
    
    sender_thread.join()
    receiver_thread.join()
    
    print("\n--- 测试总结 ---")
    if len(received_packets) == NUM_PACKETS_TO_SEND:
        print(f"✅ 成功! 发送了 {NUM_PACKETS_TO_SEND} 个包，并收到了 {len(received_packets)} 个包。")
        print("UDP通信正常。")
    elif len(received_packets) > 0:
        print(f"⚠️  部分成功。发送了 {NUM_PACKETS_TO_SEND} 个包，但只收到了 {len(received_packets)} 个。可能存在丢包。")
    else:
        print(f"❌ 失败! 发送了 {NUM_PACKETS_TO_SEND} 个包，但没有收到任何包。")
        print("请检查端口是否被占用或有防火墙规则。")