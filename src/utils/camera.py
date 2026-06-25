"""
摄像头/视频采集模块
支持：实时摄像头、视频文件、图片文件三种输入源
"""

import cv2
import numpy as np
from .config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, VIDEO_FPS


class CameraCapture:
    """统一的视频采集接口，支持摄像头和视频文件"""

    def __init__(self, source=0, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=VIDEO_FPS):
        """
        Args:
            source: 摄像头索引(int) 或 视频文件路径(str)
            width:  捕获帧宽度
            height: 捕获帧高度
            fps:    捕获帧率
        """
        self.source = source
        self.cap = None
        self.width = width
        self.height = height
        self.fps = fps
        self.is_file = isinstance(source, str)

    def start(self):
        """启动视频采集"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")

        # 设置采集参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self

    def read(self):
        """
        读取一帧
        Returns:
            ret: 是否读取成功
            frame: 图像帧 (BGR格式)
        """
        if self.cap is None:
            raise RuntimeError("请先调用 start() 启动采集")
        return self.cap.read()

    def read_with_info(self):
        """
        读取一帧并附带时间戳信息（用于测量模块）
        Returns:
            frame: 图像帧
            timestamp: 时间戳(ms)
            frame_id: 帧序号
        """
        ret, frame = self.read()
        if not ret:
            return None, None, None
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return frame, timestamp, frame_id

    def release(self):
        """释放采集资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def list_available_cameras(max_index=10):
    """
    枚举系统中可用的摄像头
    返回可用摄像头索引列表
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def resize_frame(frame, target_width=640):
    """
    等比例缩放帧（用于前端传输，减小带宽）
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def frame_to_jpeg_bytes(frame, quality=80):
    """
    将OpenCV帧编码为JPEG字节流（用于WebSocket传输）
    """
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
