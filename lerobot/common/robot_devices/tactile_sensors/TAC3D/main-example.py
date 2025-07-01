import PyTac3D
import time
import numpy as np


print('PyTac3D version is :', PyTac3D.PYTAC3D_VERSION)

# 传感器SN
# Serial Number of the Tac3D sensor
SN = ''

# 帧序号
# frame index
frameIndex = -1 


# 时间戳
# timestamp
sendTimestamp = 0.0
recvTimestamp = 0.0

# 用于存储三维形貌、三维变形场、三维分布力、三维合力、三维合力矩数据的矩阵
# Mat for storing point cloud, displacement field, distributed force, resultant force, and resultant moment
P, D, F, Fr, Mr = None, None, None, None, None


# 编写用于接收数据的回调函数
# "param"是初始化Sensor类时传入的自定义参数
# Write a callback function to receive data
# "param" is the custom parameter passed in when initializing the Sensor object
def Tac3DRecvCallback(frame, param):
    global SN, frameIndex, sendTimestamp, recvTimestamp, P, D, F, Fr, Mr
    
    # 获得传感器SN码
    SN = frame['SN']
    
    # 获得帧序号
    frameIndex = frame['index']
    
    # 获得时间戳
    sendTimestamp = frame['sendTimestamp']
    recvTimestamp = frame['recvTimestamp']

    # 获取所有数据
    P = frame.get('3D_Positions')
    D = frame.get('3D_Displacements')
    F = frame.get('3D_Forces')

    # 使用frame.get函数通过数据名称"3D_ResultantForce"获得numpy.array类型的三维合力的数据指针
    # Use the frame.get function to obtain the resultant force in the numpy.array type through the data name "3D_ResultantForce"
    Fr = frame.get('3D_ResultantForce')

    # 使用frame.get函数通过数据名称"3D_ResultantMoment"获得numpy.array类型的三维合力矩的数据指针
    # Use the frame.get function to obtain the resultant moment in the numpy.array type through the data name "3D_ResultantMoment"
    Mr = frame.get('3D_ResultantMoment')

    # 打印合力信息
    print(f"帧{frameIndex:6d} | SN:{SN} | 时间:{sendTimestamp:8.3f}s", end="")
    
    if Fr is not None and Fr.size >= 3:
        fx, fy, fz = float(Fr[0, 0]), float(Fr[0, 1]), float(Fr[0, 2])
        force_magnitude = np.sqrt(fx*fx + fy*fy + fz*fz)
        print(f" | 合力: Fx={fx:7.3f} Fy={fy:7.3f} Fz={fz:7.3f} |{force_magnitude:7.3f}|", end="")
    else:
        print(f" | 合力: N/A", end="")
    
    if Mr is not None and Mr.size >= 3:
        mx, my, mz = float(Mr[0, 0]), float(Mr[0, 1]), float(Mr[0, 2])
        moment_magnitude = np.sqrt(mx*mx + my*my + mz*mz)
        print(f" | 力矩: Mx={mx:7.3f} My={my:7.3f} Mz={mz:7.3f} |{moment_magnitude:7.3f}|")
    else:
        print(f" | 力矩: N/A")

# Create a Sensor object, set the callback function to Tac3DRecvCallback, and set the UDP receive port to 9988
print("🚀 启动 Tac3D 传感器连接...")
tac3d = PyTac3D.Sensor(recvCallback=Tac3DRecvCallback, port=9988, maxQSize=5, callbackParam='合力监控')

# 等待Tac3D传感器启动并传来数据
print("⏳ 等待 Tac3D 传感器连接...")
tac3d.waitForFrame()

print("✅ 传感器已连接，开始接收合力数据...")
print("=" * 90)
print("    帧号   |   传感器   |  时间戳  |        合力 (N)        |幅度 |        力矩 (N·m)      |幅度 |")
print("=" * 90)

# 5s
time.sleep(5)

# 发送一次校准信号
print(f"\n🔧 发送校准信号到传感器 {SN}...")
tac3d.calibrate(SN)
time.sleep(100)
print("\n选择运行模式:")
print("1. 持续运行 (按Ctrl+C停止)")
print("2. 运行指定时间")
print("3. 仅运行5秒 (默认)")

try:
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        print("🔄 持续运行模式 - 按 Ctrl+C 停止")
        print("=" * 90)
        try:
            while True:
                time.sleep(1)  # 每秒检查一次
        except KeyboardInterrupt:
            print("\n🛑 用户停止程序")
            
    elif choice == "2":
        try:
            duration = int(input("请输入运行时间（秒）: "))
            print(f"⏱️  运行 {duration} 秒")
            print("=" * 90)
            time.sleep(duration)
        except ValueError:
            print("❌ 无效输入，使用默认5秒")
            time.sleep(5)
            
    else:
        # 默认运行5秒
        print("⏱️  运行 5 秒")
        print("=" * 90)
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n🛑 用户中断程序")

# 测试手动获取帧数据
print("\n📥 测试手动获取帧数据...")
frame = tac3d.getFrame()
if not frame is None:
    print(f"✅ 手动获取帧成功，传感器SN: {frame['SN']}, 帧序号: {frame['index']}")
    
    Fr = frame.get('3D_ResultantForce')
    Mr = frame.get('3D_ResultantMoment')
    
    if Fr is not None and Fr.size >= 3:
        fx, fy, fz = float(Fr[0, 0]), float(Fr[0, 1]), float(Fr[0, 2])
        print(f"   合力: Fx={fx:.6f}, Fy={fy:.6f}, Fz={fz:.6f} N")
    else:
        print(f"   合力: N/A")
        
    if Mr is not None and Mr.size >= 3:
        mx, my, mz = float(Mr[0, 0]), float(Mr[0, 1]), float(Mr[0, 2])
        print(f"   力矩: Mx={mx:.6f}, My={my:.6f}, Mz={mz:.6f} N·m")
    else:
        print(f"   力矩: N/A")
else:
    print("❌ 缓存队列中暂无数据")

print("\n🎯 程序运行完成！")


