"""
测量服务 - 尺寸测量编排层
"""

import cv2
import time
import numpy as np
from typing import List, Dict, Tuple, Optional

from ..state import get_state
from ...measurement.dimension_measurer import DimensionMeasurer


class MeasurementService:
    """尺寸测量统一服务"""

    def __init__(self):
        pass

    def init(self) -> bool:
        """初始化测量模块"""
        try:
            measurer = DimensionMeasurer()
            from ..state import update
            update(measurer=measurer)
            print("[MeasurementService] 尺寸测量模块初始化完成")
            return True
        except Exception as e:
            print(f"[MeasurementService] 测量模块初始化失败: {e}")
            return False

    def measure(
        self,
        frame: np.ndarray,
        detection_results: List[Dict],
    ) -> Tuple[np.ndarray, List[Dict], bool]:
        """
        执行尺寸测量

        Args:
            frame: 原始帧（用于测量计算）
            detection_results: 缺陷检测结果（可能为空）
        Returns:
            processed_frame: 标注后的图像
            measurement_results: 测量结果列表
            ref_detected: 参考物是否被检测到
        """
        state = get_state()

        if not state.measurement_enabled or state.measurer is None:
            return frame, [], False

        processed_frame = frame.copy()
        measurement_results = []
        ref_detected = False

        if detection_results:
            # 有检测结果时，精确测量每个检测框
            try:
                measurements = [
                    state.measurer.measure_precise(
                        frame, d["bbox"], d.get("class", "object")
                    )
                    for d in detection_results
                ]
                for i, m in enumerate(measurements):
                    m["id"] = i + 1
                processed_frame = state.measurer.draw_measurement(processed_frame, measurements)
                measurement_results = [
                    m for m in measurements
                    if m.get("width_mm") or m.get("width_px")
                ]
            except Exception as e:
                # 单帧测量失败不影响后续帧
                pass
        else:
            # 无检测结果时，自动寻找参考物
            try:
                ref_width, frame_with_ref = state.measurer.find_reference_object(frame)
                if ref_width is not None:
                    processed_frame = frame_with_ref
                    ref_detected = True

                    h, w = frame.shape[:2]
                    measurement_results = [{
                        "label": "参考物",
                        "bbox": [w // 4, h // 4, 3 * w // 4, 3 * h // 4],
                        "width_px": round(ref_width, 1),
                        "height_px": round(ref_width, 1),
                        "width_mm": round(state.measurer.ref_width_mm, 2),
                        "height_mm": round(state.measurer.ref_width_mm, 2),
                    }]
                else:
                    h, w = frame.shape[:2]
                    cv2.putText(processed_frame, "No Reference Found | 未找到参考物",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            except Exception as e:
                # 参考物检测失败，静默继续
                pass

        return processed_frame, measurement_results, ref_detected
