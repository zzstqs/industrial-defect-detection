"""
检测服务 - 统一的缺陷检测编排层
将 YOLO 推理和传统CV检测封装为统一的服务接口
"""

import cv2
import time
import numpy as np
from typing import List, Dict, Tuple, Optional

from ..state import get_state, update
from ...detection.defect_detector import DefectDetector
from ...detection.traditional_cv import TraditionalCVDetector


class DetectionService:
    """缺陷检测统一服务"""

    def __init__(self):
        pass

    def init_yolo(self, model_path: Optional[str] = None) -> bool:
        """初始化YOLO检测器"""
        try:
            detector = DefectDetector(model_path=model_path)
            update(detector=detector)
            print(f"[DetectionService] YOLO模型加载成功")
            return True
        except Exception as e:
            print(f"[DetectionService] YOLO模型加载失败: {e}")
            update(detector=None)
            return False

    def init_traditional(self) -> bool:
        """初始化传统CV检测器"""
        try:
            trad = TraditionalCVDetector()
            update(trad_cv_detector=trad)
            print("[DetectionService] 传统CV检测器初始化完成")
            return True
        except Exception as e:
            print(f"[DetectionService] 传统CV初始化失败: {e}")
            update(trad_cv_detector=None)
            return False

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        执行缺陷检测

        Args:
            frame: BGR 图像帧

        Returns:
            processed_frame: 标注后的图像
            detections: 检测结果列表
        """
        state = get_state()
        detection_results = []

        if state.detection_enabled:
            if state.detection_method == "yolo":
                detection_results = self._detect_yolo(frame)
            else:
                detection_results = self._detect_traditional(frame)

        return frame, detection_results

    def process_and_annotate(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """一站式检测+标注"""
        t0 = time.time()
        state = get_state()
        processed_frame = frame.copy()
        detection_results = []

        if not state.detection_enabled:
            return processed_frame, detection_results

        if state.detection_method == "yolo" and state.detector is not None:
            has_model = hasattr(state.detector, 'model') and state.detector.model is not None
            if has_model:
                processed_frame, detection_results = state.detector.process_frame(frame)
            else:
                # YOLO不可用时降级到传统CV
                processed_frame, detection_results = self._detect_traditional(processed_frame)

        elif state.trad_cv_detector is not None:
            processed_frame, detection_results = self._detect_traditional(processed_frame)

        # 无检测结果时的提示
        if not detection_results:
            processed_frame = draw_no_defects(processed_frame)

        # 更新统计
        elapsed = (time.time() - t0) * 1000
        n_frames = state.total_frames + 1
        update(
            total_frames=n_frames,
            total_detections=state.total_detections + len(detection_results),
            avg_processing_ms=(
                state.avg_processing_ms * state.total_frames + elapsed
            ) / n_frames,
        )

        return processed_frame, detection_results

    def _detect_yolo(self, frame: np.ndarray) -> List[Dict]:
        """YOLO检测"""
        state = get_state()
        if state.detector is None:
            return []
        try:
            results = state.detector.predict(frame)
            _, detections = state.detector.draw_results(frame, results)
            return detections
        except Exception as e:
            print(f"[DetectionService] YOLO推理错误: {e}")
            return []

    def _detect_traditional(self, frame: np.ndarray) -> List[Dict]:
        """传统CV检测"""
        state = get_state()
        if state.trad_cv_detector is None:
            return []

        try:
            processed_frame, raw_detections = state.trad_cv_detector.detect(frame)

            # 适配格式
            return [
                {
                    "id": i + 1,
                    "class": r.get("method", "defect"),
                    "confidence": round(r.get("confidence", 0.5), 3),
                    "bbox": r["bbox"],
                    "width": r.get("width", r["bbox"][2] - r["bbox"][0]),
                    "height": r.get("height", r["bbox"][3] - r["bbox"][1]),
                }
                for i, r in enumerate(raw_detections)
            ]
        except Exception as e:
            print(f"[DetectionService] 传统CV检测错误: {e}")
            return []


def draw_no_defects(frame: np.ndarray) -> np.ndarray:
    """在帧底部绘制'未检测到缺陷'提示"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    banner_h = 40
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "No Defects Detected | 未检测到缺陷",
                (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    return frame
