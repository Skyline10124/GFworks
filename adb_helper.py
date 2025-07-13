from ppadb.client import Client as AdbClient
from adb_shell.adb_device import AdbDeviceTcp
from collections import deque
import subprocess
import time
import socket
import platform
import datetime
import os
import cv2
import numpy as np
import queue
import threading

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
        
    def click(self, x, y):
        # 在设备上模拟点击操作
        if not self.device:
            print("设备未连接")
            return False
        command = f"input tap {x} {y}"
        try:
            self.execute_command(command)
            return True
        except Exception as e:
            print(f"点击操作失败: {str(e)}")
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
        
class OCRProcessor:
    # OCR处理器类，负责图像中的文本识别
    def __init__(self,lang='ch'):
        self.lang = lang
        self.ocr_available = False
        self.ocr_model = None

        # 尝试导入PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.PaddleOCR = PaddleOCR
            self.ocr_available = True
            print("PaddleOCR已成功导入")
        except ImportError:
            print("PaddleOCR未找到，请安装PaddleOCR")

    def initialize_ocr(self):
        # 初始化OCR模型
        if not self.ocr_available:
            return False
        
        if self.ocr_model is None:
            return True
        
        try:
            # 若GPU可用，则使用GPU
            use_gpu = False
            try:
                import paddle
                if paddle.device.is_compiled_with_cuda():
                    use_gpu = True
            except:
                pass

            # 初始化OCR模型
            self.ocr_model = self.PaddleOCR(
                lang=self.lang, 
                use_angle_cls=True, 
                use_gpu=use_gpu,
                show_log=False
                )
            print(f"PaddleOCR 模型已初始化 (使用 {'GPU' if use_gpu else 'CPU'})")
            return True
        except Exception as e:
            print(f"PaddleOCR 模型初始化失败：{e}")
            self.ocr_available = False
            return False

    def processing_img(self,img):
        # 处理图像，进行OCR识别
        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 自适应阈值二值化
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # 降噪
        kernel = np.ones((1,1),np.uint8)
        denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return denoised
    
    def detect_text(self,img):
        # 使用PaddleOCR检测图像中的文本
        if not self.ocr_available:
            print("PaddleOCR未可用")
            return []
        
        # 延迟初始化模型
        if self.ocr_model is None:
            if not self.ocr_available:
                return []
            
        try:
            # 执行OCR识别
            result = self.ocr_model.ocr(img,cls=True)

            text_boxes = []
            for line in result:
                if line is None:
                    continue
                for word_info in line:
                    if word_info is None:
                        continue

                    # 提取文本信息
                    point = word_info[0]
                    text = word_info[1][0]
                    confidence = word_info[1][1]

                    # 保留置信度大于0.6的结果
                    if confidence < 0.6:
                        continue

                    # 计算边界框
                    xs = [p[0] for p in point]
                    ys = [p[1] for p in point]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    width = x_max - x_min
                    height = y_max - y_min

                    # 计算中心点
                    center_x = int((x_min + x_max) / 2)
                    center_y = int((y_min + y_max) / 2)

                    text_boxes.append({
                        "text": text,
                        'box': (int(x_min), int(y_min), int(width), int(height)),
                        'center': (center_x, center_y)
                    })

            return text_boxes
    
        except Exception as e:
            print(f"OCR处理失败: {str(e)}")
            return []
        
    def check_ocr_status(self):
        """检查 OCR 状态并返回详细信息"""
        status = {
            'available': self.ocr_available,
            'initialized': self.ocr_model is not None,
            'language': self.lang
        }
        
        if self.ocr_available and self.ocr_model is not None:
            try:
                import paddle
                status['gpu_supported'] = paddle.device.is_compiled_with_cuda()
                status['gpu_enabled'] = self.ocr_model.use_gpu
            except:
                status['gpu_supported'] = False
                status['gpu_enabled'] = False
        
        return status

class ScreenshotBuffer:
    def __init__(self, adb_helper, buffer_seconds=10, fps=10, max_works=4):
        # 初始化截图缓冲区
        self.adb = adb_helper
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_works = max_works
        self.frame_interval = 1 / fps

        # 循环缓存区存储最近N秒的截图
        self.buffer_size = buffer_seconds * fps
        self.screenshot_buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()

        # 工作线程控制
        self.capturing = False
        self.processing = False
        self.capture_thread = None
        self.process_thread = []

        # 添加处理队列和节流控制
        self.process_queue = queue.Queue(maxsize=10)  # 限制队列大小
        self.last_process_time = 0
        self.process_interval = 0.1  # 最小处理间隔 (秒)
        self.max_queue_size = 5      # 最大待处理帧数
        
        # 添加性能计数器
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = time.time()

        # OCR处理
        self.ocr_processor = OCRProcessor()

    def start(self):
        # 启动截图和处理线程
        if self.capturing:
            return
        
        self.capturing = True
        self.processing = True

        # 启动截图处理线程
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # 启动处理线程
        for i in range(self.max_works):
            t = threading.Thread(target=self.process_loop)
            t.daemon = True
            t.start()
            self.process_thread.append(t)

        print(f"截图处理已启动{self.fps}FPS，缓存{self.buffer_seconds}秒，当前最大处理线程数：{self.max_works}")

    def stop(self):
        # 停止截图和处理线程
        self.capturing = False
        self.processing = False

        if self.capture_thread:
            self.capture_thread.join(timeout=1)

        for t in self.process_threads:
            t.join(timeout=0.5)

        print("截图处理已停止")

    def capture_loop(self):
        # 循环捕获屏幕截图
        last_time = time.time()

        while self.capturing:
            start_time = time.time()

            try:
                # 截图并保存
                screenshot_data = self.adb.device.screencap()
                if screenshot_data:
                    # 添加时间戳
                    timestamp = time.time()

                    # 添加到缓冲区
                    with self.lock:
                        self.screenshot_buffer.append((timestamp, screenshot_data))

                    if self.process_queue.qsize() < self.max_queue_size:
                        self.process_queue.put((timestamp, screenshot_data))

                    self.frame_count += 1

            except Exception as e:
                print(f"截图失败: {str(e)}")

            # 计算下一帧的时间
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_interval - elapsed)

            # 确保精确的帧率
            if sleep_time > 0:
                time.sleep(sleep_time)

            # 定期打印性能报告
            current_time = time.time()
            if current_time - last_time > 5:
                last_print_time = current_time
                total_time = current_time - self.start_time
                fps = self.frame_count / total_time if total_time > 0 else 0
                queue_size = self.process_queue.qsize()
                print(f"性能报告: 截图FPS={fps:.1f}, 待处理帧={queue_size}/{self.max_queue_size}")

            # 调试信息
            if time.time() - last_time > 5:
                last_time = time.time()
                print(f"缓冲区状态：{len(self.screenshot_buffer)}/{self.buffer_size}帧，最近截图时间：{self.screenshot_buffer[-1][0] if self.screenshot_buffer else '无'}")

    def process_loop(self):
        """处理循环（优化CPU使用）"""
        while self.processing:
            try:
                # 从队列获取截图（带超时）
                try:
                    timestamp, screenshot_data = self.process_queue.get(timeout=0.5)
                except queue.Empty:
                    continue  # 队列为空时跳过
                
                # 处理节流：确保最小处理间隔
                current_time = time.time()
                if current_time - self.last_process_time < self.process_interval:
                    time.sleep(self.process_interval - (current_time - self.last_process_time))
                
                # 处理截图
                self.process_screenshot(screenshot_data)
                self.processed_count += 1
                self.last_process_time = time.time()
                
                # 更新队列状态
                self.process_queue.task_done()
            
            except Exception as e:
                print(f"处理线程错误: {str(e)}")

    def process_screenshot(self, screenshot_data):
        """优化处理流程"""
        # 转为OpenCV图像格式
        img = cv2.imdecode(np.frombuffer(screenshot_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        
        # 仅在需要时进行OCR处理
        if self.needs_ocr_processing(img):
            # 确保 OCR 处理器已初始化
            if not self.ocr_processor.ocr_available:
                print("警告: OCR 功能不可用，跳过处理")
                return
                
            if self.ocr_processor.ocr_model is None:
                print("初始化 OCR 模型...")
                if not self.ocr_processor.initialize_ocr():
                    print("OCR 初始化失败，跳过处理")
                    return
            
            text_boxes = self.ocr_processor.detect_text(img)
            # 使用OCR结果...
        
        # 特征识别（优化为按需执行）
        if self.needs_feature_detection(img):
            features = self.detect_features(img)
            # 使用特征结果...

    def detect_features(self, img):
        # 识别图像中的特征
        # 转为HSV颜色空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 检测红色按钮（示例）
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        mask = mask1 + mask2
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {}
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 判断是否是按钮
                if w > 50 and h > 50 and w/h > 0.8 and w/h < 1.2:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    features["start_button"] = (center_x, center_y)
        
        return features
    
    def get_latest_frame(self):
        # 获取最新的截图帧
        with self.lock:
            if not self.screenshot_buffer:
                return None
            return self.screenshot_buffer[-1]
        
    def get_frame_in_range(self, start_time, end_time):
        # 获取指定时间范围内的截图帧
        with self.lock:
            frames = []
            for timestamp, screenshot_data in self.screenshot_buffer:
                if start_time <= timestamp <= end_time:
                    frames.append((timestamp, screenshot_data))
            return frames
        
    def needs_ocr_processing(self, img):
        """判断是否需要OCR处理（减少不必要的OCR调用）"""
        # 示例逻辑：仅当图像包含文字区域时才进行OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果有足够大的轮廓，可能包含文本
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # 面积阈值
                return True
        return False
    
    def needs_feature_detection(self, img):
        """判断是否需要特征检测（减少不必要的计算）"""
        # 示例逻辑：仅当画面变化超过阈值时才检测
        if not hasattr(self, 'last_frame'):
            self.last_frame = img
            return True
        
        # 计算帧间差异
        diff = cv2.absdiff(img, self.last_frame)
        diff_mean = np.mean(diff)
        self.last_frame = img
        
        return diff_mean > 10  # 差异阈值

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