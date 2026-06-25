"""
项目配置文件
集中管理所有可配置参数，方便调整和答辩展示
"""

import os

# ============ 项目路径 ============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")

# ============ 相机/视频配置 ============
CAMERA_INDEX = 0                  # 摄像头索引，0为默认摄像头
VIDEO_FPS = 30                    # 视频帧率
FRAME_WIDTH = 1280                # 帧宽度
FRAME_HEIGHT = 720                # 帧高度

# ============ 缺陷检测配置（YOLO） ============
YOLO_MODEL_NAME = "yolo11n.pt"  # 使用的YOLO模型（注意：v11 命名为 yolo11，不是 yolov11）
TRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")  # 训练好的模型路径
CONF_THRESHOLD = 0.15             # 置信度阈值（demo降低以便更容易检测到）
IOU_THRESHOLD = 0.45              # IOU阈值（NMS用）
DEFECT_CLASSES = ["scratch", "crack", "stain", "dent", "corrosion"]  # 自定义缺陷类别（与训练一致）

# ============ 尺寸测量配置 ============
# 相机标定参数（需实际标定后填入）
# 这里提供示例值，实际项目需要标定
CAMERA_MATRIX = None              # 3x3内参矩阵，标定后填入
DIST_COEFFS = None                # 畸变系数，标定后填入
REAL_UNIT = "mm"                  # 测量单位

# 参考物体参数（用于无标定时的粗略测量）
REF_OBJECT_WIDTH_MM = 25.0        # 参考物体实际宽度(mm)，默认硬币
REF_OBJECT_TYPE = "coin"          # 参考物类型: "coin"(硬币25mm), "a4"(A4纸210mm), "custom"

# ============ 后端服务配置 ============
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
WEBSOCKET_PATH = "/ws/video"

# ============ 前端配置 ============
FRONTEND_TITLE = "工业视觉检测系统"
DETECTION_COLOR = (0, 255, 0)   # 缺陷框颜色 (BGR)
MEASUREMENT_COLOR = (255, 0, 0) # 测量线颜色 (BGR)
TEXT_COLOR = (255, 255, 255)    # 文字颜色
