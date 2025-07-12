import cv2
import numpy as np
from adb_helper import ADBHelper, ADBScreenshotHelper

print("hello world")

adb = ADBHelper()
devices = adb.list_devices()

if adb.connect(ip='127.0.0.1',port=16448):
    print("成功连接到设备")

    # 创建截图助手
    screenshot_helper = ADBScreenshotHelper(adb)

    # 截图并保存
    if screenshot_helper.capture_screenshot():
        # 保存原始截图
        screenshot_helper.save_screenshot()

        # 使用OpenCV处理截图
        processed_path = screenshot_helper.process_with_opencv()
        
        # 显示处理后的图像
        if processed_path:
            screenshot_helper.display_processed_image()