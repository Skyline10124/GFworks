import cv2
import numpy as np
import time
import threading
from adb_helper import ADBHelper, ScreenshotBuffer, OCRProcessor

print("启动高性能截图处理系统")

# 创建ADBHelper实例
adb = ADBHelper()

# 在启动截图系统前添加OCR测试
print("测试PaddleOCR功能...")

# 创建OCR处理器
ocr = OCRProcessor()

# 创建一个测试图像
test_img = np.zeros((100, 300, 3), dtype=np.uint8)
cv2.putText(test_img, "OCR测试", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 尝试OCR识别
text_boxes = ocr.detect_text(test_img)
if text_boxes:
    print("PaddleOCR测试成功! 识别结果:")
    for box in text_boxes:
        print(f"- 文本: '{box['text']}', 位置: {box['box']}")
else:
    print("PaddleOCR测试失败! 请检查安装配置")

# 连接到设备
if adb.connect(ip='127.0.0.1', port=16448):
    print("成功连接到设备")
    
    # 创建截图缓存系统
    screenshot_buffer = ScreenshotBuffer(
        adb, 
        buffer_seconds=10, 
        fps=10,
        max_works=4
    )
    
# 在启动系统前添加 OCR 状态检查
print("检查 OCR 状态...")
ocr_status = screenshot_buffer.ocr_processor.check_ocr_status()
print(f"OCR 可用: {ocr_status['available']}")
print(f"OCR 已初始化: {ocr_status['initialized']}")
print(f"使用语言: {ocr_status['language']}")

if ocr_status['available']:
    if not ocr_status['initialized']:
        print("正在初始化 OCR 模型...")
        if screenshot_buffer.ocr_processor.initialize_ocr():
            print("OCR 初始化成功")
        else:
            print("OCR 初始化失败")
    
    if ocr_status.get('gpu_supported', False):
        print(f"GPU 支持: {'已启用' if ocr_status['gpu_enabled'] else '已禁用'}")
    else:
        print("GPU 支持: 不可用")
else:
    print("OCR 功能不可用，请检查 PaddleOCR 安装")

# 启动系统
screenshot_buffer.start()

# 添加性能监控线程
def performance_monitor(buffer):
    """独立线程监控系统性能"""
    while buffer.processing:
        time.sleep(5)
        total_time = time.time() - buffer.start_time
        fps = buffer.frame_count / total_time if total_time > 0 else 0
        process_rate = buffer.processed_count / total_time if total_time > 0 else 0
        queue_size = buffer.process_queue.qsize()
        
        print(f"\n[性能监控] 截图FPS: {fps:.1f} | 处理FPS: {process_rate:.1f} | 队列: {queue_size}/{buffer.max_queue_size}")
        print(f"[资源使用] 截图线程: {'活跃' if buffer.capture_thread.is_alive() else '休眠'} | 处理线程: {len(buffer.process_thread)}个")

# 在启动系统后添加性能监控
if adb.connect(ip='127.0.0.1', port=16448):
    print("成功连接到设备")
    
    # 创建截图缓存系统
    screenshot_buffer = ScreenshotBuffer(
        adb, 
        buffer_seconds=30, 
        fps=30,
        max_works=4
    )
    
    # 启动系统
    screenshot_buffer.start()
    
    # 启动性能监控线程
    monitor_thread = threading.Thread(
        target=performance_monitor, 
        args=(screenshot_buffer,),
        daemon=True
    )
    monitor_thread.start()
    
    try:
        # 主循环
        while True:
            # 获取最新帧
            latest_frame = screenshot_buffer.get_latest_frame()
            if latest_frame:
                timestamp, screenshot_data = latest_frame
                
                # 显示帧率
                print(f"当前帧率: {1/(time.time()-timestamp):.1f}FPS" if time.time()-timestamp > 0 else "初始化中...")
                
                # 转换为OpenCV图像
                img = cv2.imdecode(np.frombuffer(screenshot_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                # 显示图像（可选）
                cv2.imshow("实时预览", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        # 停止系统
        screenshot_buffer.stop()
        cv2.destroyAllWindows()