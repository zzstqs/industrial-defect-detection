"""
应用全局状态管理
统一管理检测/测量/摄像头等全局状态，避免散落的全局变量
提供线程安全的状态读写接口
"""

import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class AppState:
    """应用全局状态容器"""

    # === 核心状态 ===
    detection_enabled: bool = False
    measurement_enabled: bool = False
    detection_method: str = "yolo"       # "yolo" | "traditional"
    measurement_mode: str = "reference"  # "reference" | "calibrated"

    # === 检测/测量参数 ===
    conf_threshold: float = 0.15
    iou_threshold: float = 0.45
    max_detections: int = 50

    # === 摄像头状态 ===
    camera_active: bool = False
    camera_source: int = 0

    # === 运行时统计 ===
    total_frames: int = 0
    total_detections: int = 0
    last_frame_time: float = 0.0
    avg_processing_ms: float = 0.0

    # === 模块引用 (运行时注入) ===
    detector: Any = None          # DefectDetector
    trad_cv_detector: Any = None  # TraditionalCVDetector
    measurer: Any = None          # DimensionMeasurer
    stabilizer: Any = None        # DetectionStabilizer


# 全局单例
_state = AppState()
_lock = threading.RLock()


def get_state() -> AppState:
    """获取全局状态对象（非线程安全读取，仅用于读）"""
    return _state


def update(**kwargs) -> AppState:
    """线程安全地更新状态"""
    with _lock:
        for key, value in kwargs.items():
            if hasattr(_state, key):
                setattr(_state, key, value)
    return _state


def snapshot() -> Dict[str, Any]:
    """获取状态快照（用于API返回）"""
    return {
        "detection_enabled": _state.detection_enabled,
        "measurement_enabled": _state.measurement_enabled,
        "detection_method": _state.detection_method,
        "measurement_mode": _state.measurement_mode,
        "camera_active": _state.camera_active,
        "total_frames": _state.total_frames,
        "total_detections": _state.total_detections,
        "avg_processing_ms": round(_state.avg_processing_ms, 1),
    }


def health_check() -> Dict[str, Any]:
    """系统健康检查"""
    return {
        "status": "ok",
        "detection_enabled": _state.detection_enabled,
        "measurement_enabled": _state.measurement_enabled,
        "detection_method": _state.detection_method,
        "camera_active": _state.camera_active,
        "yolo_available": (
            _state.detector is not None
            and hasattr(_state.detector, 'model')
            and _state.detector.model is not None
        ),
        "trad_cv_available": _state.trad_cv_detector is not None,
        "measurer_available": _state.measurer is not None,
        "stabilizer_active": _state.stabilizer is not None,
        "total_frames": _state.total_frames,
        "avg_processing_ms": round(_state.avg_processing_ms, 1),
    }
