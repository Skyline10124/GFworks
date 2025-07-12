import cv2
import numpy as np
from adb_helper import ADBHelper

print("hello world")

adb = ADBHelper()
def test_adb_connection():
    # 测试ADB连接
    assert adb.connect() is True
    print("已连接到设备", adb.execute_command("getprop ro.product.model"))

def test_adb_command_execution():
    # 测试ADB命令执行
    assert adb.connect() is True
    result = adb.execute_command("echo 'Hello ADB'")
    assert "Hello ADB" in result

if __name__ == "__main__":
    test_adb_connection()
    test_adb_command_execution()
    print("所有测试通过")