from ppadb.client import Client as AdbClient
from adb_shell.adb_device import AdbDeviceTcp
import subprocess
import time
import socket
import platform
import datetime
import os

class ADBHelper:
    def __init__(self, host='127.0.0.1',port=5037):
        # 初始化adb客户端，连接到设备5037端口
        self.client = AdbClient(host=host, port=port)
        self.device = None
        self.devices = []
    
    def connect(self,serial=None,ip=None,port=None,max_retries=3,retry_interval=1):
        # 连接到指定设备，未指定则连接到默认设备，支持AdbClient和TCP连接，支持重试机制
        # 优先使用TCP连接
        if ip and port:
            return self.connect_tcp(ip, port, max_retries, retry_interval)
        
        # 尝试AdbClient连接
        for attempt in range(max_retries):
            try:
                self.devices = self.client.devices()
                if not self.devices:
                    print(f"尝试{attempt + 1}/{max_retries}，未找到任何设备")
                    continue
            
                # 指定设备序列号连接
                if serial:
                    for dev in self.devices:
                        if dev.serial == serial:
                            self.device = dev
                            print(f"尝试{attempt + 1}/{max_retries}，已连接到设备: {dev.serial}")
                            return True
                    print(f"尝试{attempt + 1}/{max_retries}，未找到序列号为 {serial} 的设备")
            
                # 默认连接第一个设备
                else:
                    self.device = self.devices[0]
                    print(f"尝试{attempt + 1}/{max_retries}，已连接到设备: {self.device.serial}")
                    return True
            except RuntimeError as e:
                print(f"尝试{attempt + 1}/{max_retries}，连接失败: {str(e)}")

            # 等待1秒后重试
            time.sleep(1)

        print(f"连接到设备失败，已重试 {max_retries} 次")
        return False
        
    def connect_tcp(self,ip='127.0.0.1',port=16384,max_retries=3,retry_interval=1):
        # 通过TCP连接到设备，支持重试机制
        # 检查端口是否可用
        if not self.is_port_available(ip, port):
            print(f"端口 {port} 已被占用，尝试释放端口")
            self.release_port(port)

        serial = f"{ip}:{port}"

        for attempt in range(max_retries):
            try:
                # 构建adb connect命令
                command = f"adb connect {ip}:{port}"
                result = subprocess.run(command,shell=True,capture_output=True,timeout=10)

                # 安全处理输出（解决Unicode编码问题）
                stdout = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ""
                stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ""
    
                # 修复NoneType错误
                if "connected" in stdout:
                    print(f"成功连接到设备 {ip}:{port}(尝试 {attempt + 1}/{max_retries})")

                    # 更新设备列表
                    self.devices = self.client.devices()

                    # 查找serial为ip:port的设备
                    for dev in self.devices:
                        if dev.serial == serial:
                            self.device = dev
                            print(f"尝试 {attempt+1}/{max_retries}，已连接到设备: {serial}")
                            return True
                        
                    print(f"连接成功，但未在设备列表中找到{serial}的设备")
                else:
                    error_msg = stdout + stderr
                    print(f"连接失败(尝试 {attempt + 1}/{max_retries})")
                    print(f"错误信息: {error_msg}")

            except Exception as e:
                print(f"尝试 {attempt + 1}/{max_retries} 连接失败")
                print(f"错误信息: {str(e)}")
            
            # 如果不是最后一次，则等待
            if attempt < max_retries - 1:
                time.sleep(retry_interval)

        print(f"TCP连接到设备 {ip}:{port}失败，已重试 {max_retries} 次")
        return False

    def list_devices(self):
        # 列出所有连接的设备
        try:
            self.devices = self.client.devices()
            if not self.devices:
                print("未找到任何设备")
                return []
            
            device_info = []
            for i, dev in enumerate(self.devices):
                device_info.append({
                    "index": i,
                    "serial": dev.serial,
                    "model": dev.get_properties().get("ro.product.model", "未知型号"),
                    "state": dev.get_properties().get("sys.boot_completed", "未知状态")
                })
            return device_info
        except RuntimeError as e:
            print(f"列出设备失败: {str(e)}")
            return []

    def execute_command(self, command):
        # 在设备上执行adb shell命令
        if not self.device:
            print("设备未连接")
            return None
        return self.device.shell(command)

    def is_port_available(self, ip, port):
        # 检查指定端口是否可用
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                return sock.connect_ex((ip, port)) != 0
        except Exception as e:
            print(f"检查端口 {port} 可用性失败: {str(e)}")
            return False
        
    def release_port(self, port):
        # 释放指定端口
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    f"netstat -ano | findstr :{port}",
                    shell=True, capture_output=True, text=True
                )
                if result.stdout:
                    pid = result.stdout.split()[-1]
                    subprocess.run(f"taskkill /PID {pid} /F", shell=True)
                    print(f"已释放端口 {port}")
                else:
                    result = subprocess.run(
                        f"lsof -i :{port}",
                        shell=True, capture_output=True, text=True
                    )
                    if result.stdout:
                        pid = result.stdout.split()[1]
                        subprocess.run(f"kill -9 {pid}", shell=True)
                        print(f"已释放端口 {port}")
                    else:
                        print(f"端口 {port} 未被占用")
                print(f"Windows系统，已释放端口 {port}")
                return True
        except Exception as e:
            print(f"释放端口 {port} 失败: {str(e)}")
            return False
        
class ADBScreenshotHelper:
    def __init__(self, adb_helper):
        # 初始化ADB截图助手
        self.adb = adb_helper
        self.screenshot_data = None
        self.processed_image = None
    
    def capture_screenshot(self):
        # 捕获设备屏幕截图
        if not self.adb.device:
            print("设备未连接")
            return False
        
        try:
            self.screenshot_data = self.adb.device.screencap()
            return True
        except Exception as e:
            print(f"捕获屏幕截图失败: {str(e)}")
            return False
        
    def save_screenshot(self,output_dir=".\data\screenshots",filename=None):
        # 保存屏幕截图到指定目录
        if not self.screenshot_data:
            print("没有捕获到屏幕截图")
            return None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        if not filename:
            filestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{filestamp}.png"

        filepath = os.path.join(output_dir, filename)

        # 保存截图
        try:
            with open(filepath, "wb") as f:
                f.write(self.screenshot_data)
            print(f"截图已保存到: {filepath}")
            return filepath
        except Exception as e:
            print(f"保存截图失败: {str(e)}")
            return None
        
'''
# 实例用法
adb = ADBHelper()
if adb.connect():
    print("已连接默认设备:", adb.device.serial)

# 2. 通过序列号连接指定设备（USB连接）
if adb.connect(serial="ABCDEF123456"):
    print("已连接指定设备:", adb.device.serial)

# 3. 通过TCP连接设备
if adb.connect(ip="192.168.1.100", port=5555):
    print("已通过TCP连接设备:", adb.device.serial)

# 4. 连接Mumu模拟器（默认端口16384）
if adb.connect(ip="127.0.0.1", port=16384):
    print("已连接Mumu模拟器:", adb.device.serial)
'''