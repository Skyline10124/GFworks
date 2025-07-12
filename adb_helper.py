from ppadb.client import Client as AdbClient
from adb_shell.adb_device import AdbDeviceTcp

class ADBHelper:
    def __init__(self, host='127.0.0.1',port=5037):
        # 初始化adb客户端，连接到设备5037端口
        self.client = AdbClient(host=host, port=port)
        self.device = None
    
    def connect(self):
        # 连接到adb服务器并获取第一个设备
        try:
            devices = self.client.devices()
            if devices:
                self.device = devices[0]
                print("已连接设备：", self.device.serial)
                return True
            print("未找到设备")
            return False
        except RuntimeError as e:
            print(f"连接设备失败: {str(e)}")
            return False
        
    def execute_command(self, command):
        # 在设备上执行adb shell命令
        if not self.device:
            print("设备未连接")
            return None
        return self.device.shell(command)
    
# 实例用法
if __name__ == "__main__":
    adb = ADBHelper()
    if adb.connect():
        print("设备信息", adb.execute_command("getprop ro.product.model"))