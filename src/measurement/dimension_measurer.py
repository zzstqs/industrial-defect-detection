"""
尺寸测量模块 - 基于 OpenCV（增强版）
支持：相机标定测量、参考物比例测量、亚像素边缘检测、精确轮廓拟合、正交测量、实时视频流测量
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from ..utils.config import CAMERA_MATRIX, DIST_COEFFS, REF_OBJECT_WIDTH_MM, REAL_UNIT, REF_OBJECT_TYPE


# ============ 参数配置 ============

@dataclass
class MeasurementParams:
    """尺寸测量可调参数"""
    # 参考物检测
    circle_param1: int = 100       # HoughCircles param1
    circle_param2: int = 30        # HoughCircles param2
    circle_min_r: int = 20         # 最小圆半径
    circle_max_r: int = 200        # 最大圆半径

    # 亚像素边缘
    use_subpixel: bool = True      # 是否使用亚像素精度
    canny_low: int = 50            # Canny 低阈值
    canny_high: int = 150          # Canny 高阈值

    # 轮廓筛选
    contour_area_min: int = 100    # 最小轮廓面积
    contour_area_max: int = 500000 # 最大轮廓面积

    # 显示
    show_measure_lines: bool = True  # 显示测量辅助线


# ============ 亚像素工具函数 ============

def subpixel_edge(gray: np.ndarray, roi: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
    """
    亚像素边缘检测 (基于 Canny + 梯度插值)
    Args:
        gray: 灰度图
        roi:  (x1, y1, x2, y2) 感兴趣区域
    Returns:
        亚像素边缘点列表 [(x, y), ...]
    """
    x1, y1, x2, y2 = roi
    x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
    x2, y2 = min(gray.shape[1], x2 + 5), min(gray.shape[0], y2 + 5)

    roi_gray = gray[y1:y2, x1:x2]
    if roi_gray.size == 0:
        return []

    # Canny 边缘
    edges = cv2.Canny(roi_gray, 50, 150)

    # 找边缘像素位置
    edge_pts = np.argwhere(edges > 0)

    # 用 Sobel 梯度做亚像素插值
    grad_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)

    subpixel_pts = []
    for py, px in edge_pts:
        if 1 <= py < roi_gray.shape[0] - 1 and 1 <= px < roi_gray.shape[1] - 1:
            gx = grad_x[py, px]
            gy = grad_y[py, px]
            grad_mag = math.sqrt(gx * gx + gy * gy) + 1e-9

            # 梯度方向上的亚像素偏移
            nx, ny = gx / grad_mag, gy / grad_mag
            sub_x = px + nx * 0.5 + x1
            sub_y = py + ny * 0.5 + y1
            subpixel_pts.append((sub_x, sub_y))

    return subpixel_pts


# ============ 主测量类 ============

class DimensionMeasurer:
    """
    工业尺寸测量仪（增强版）
    支持两种模式：
    1. 标定模式（精确）：需要相机标定参数
    2. 参考物模式（便捷）：放置已知尺寸参考物，自动计算比例

    增强功能：
    - 亚像素边缘检测
    - 精确轮廓拟合 (最小外接矩形/圆/椭圆)
    - 正交尺寸测量
    - 轮廓面积、周长、圆度、长宽比等几何特征
    """

    def __init__(self, camera_matrix=None, dist_coeffs=None, ref_width_mm=None, ref_type="coin"):
        """
        Args:
            camera_matrix: 3x3相机内参矩阵（标定模式）
            dist_coeffs:  畸变系数（标定模式）
            ref_width_mm:  参考物实际宽度(mm)（参考物模式）
            ref_type:      参考物类型 "coin"(25mm) / "a4"(210mm) / "custom"
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.is_calibrated = (camera_matrix is not None)

        # 参考物模式参数
        self.ref_type = ref_type
        if ref_type == "coin":
            self.ref_width_mm = 25.0   # 一元硬币直径
        elif ref_type == "a4":
            self.ref_width_mm = 210.0  # A4纸宽度
        elif ref_type == "custom":
            self.ref_width_mm = ref_width_mm or REF_OBJECT_WIDTH_MM
        else:
            self.ref_width_mm = ref_width_mm or REF_OBJECT_WIDTH_MM

        self.pixel_per_mm = None       # 像素/实际mm
        self.last_measurements = []
        self.params = MeasurementParams()

    def set_pixel_per_mm(self, ref_pixel_width, ref_actual_width_mm=None):
        """设置像素比例"""
        actual_width = ref_actual_width_mm or self.ref_width_mm
        self.pixel_per_mm = ref_pixel_width / actual_width

    def px_to_mm(self, px_val: float) -> Optional[float]:
        """像素值转实际毫米"""
        return round(px_val / self.pixel_per_mm, 2) if self.pixel_per_mm else None

    def measure_distance(self, pt1, pt2):
        """测量两点间距离"""
        dist_px = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        return dist_px, self.px_to_mm(dist_px)

    def measure_precise(self, frame: np.ndarray, bbox: List[int], label: str = "") -> Dict:
        """
        精确测量一个区域（含亚像素边界拟合）
        Args:
            frame: 输入图像 (BGR)
            bbox:  [x1, y1, x2, y2]
            label: 标签
        Returns:
            包含几何特征的完整字典
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1) - 5), max(0, int(y1) - 5)
        x2, y2 = min(frame.shape[1], int(x2) + 5), min(frame.shape[0], int(y2) + 5)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return self._empty_result(label, bbox)

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Canny 边缘检测
        edges = cv2.Canny(roi_gray, self.params.canny_low, self.params.canny_high)

        # 形态学连接
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self._empty_result(label, bbox)

        # 取最大轮廓（主物体）
        main_cnt = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(main_cnt)

        if area_px < self.params.contour_area_min:
            return self._empty_result(label, bbox)

        # 偏移回原图坐标
        main_cnt_offset = main_cnt + np.array([x1, y1])

        # 几何特征
        perimeter = cv2.arcLength(main_cnt, True)
        circularity = 4 * np.pi * area_px / (perimeter * perimeter + 1e-6)

        # 最小外接矩形
        rect = cv2.minAreaRect(main_cnt_offset)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        rect_w, rect_h = rect[1]

        # 最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(main_cnt_offset)

        # 凸包
        hull = cv2.convexHull(main_cnt_offset)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / (hull_area + 1e-6)

        # 椭圆拟合
        ellipse = None
        if len(main_cnt) >= 5:
            ellipse = cv2.fitEllipse(main_cnt_offset)

        # 正交尺寸（用旋转矩形）
        ortho_w = max(rect_w, rect_h)
        ortho_h = min(rect_w, rect_h)
        aspect_ratio = ortho_w / (ortho_h + 1e-6)

        return {
            "label": label,
            "bbox": bbox,
            "width_px": int(ortho_w),
            "height_px": int(ortho_h),
            "width_mm": self.px_to_mm(ortho_w),
            "height_mm": self.px_to_mm(ortho_h),
            "area_px": int(area_px),
            "area_mm2": round(self.px_to_mm(area_px) * 2, 2) if self.pixel_per_mm else None,
            "perimeter_px": int(perimeter),
            "perimeter_mm": self.px_to_mm(perimeter),
            "circularity": round(circularity, 3),
            "solidity": round(solidity, 3),
            "aspect_ratio": round(aspect_ratio, 3),
            "radius_px": int(radius),
            "radius_mm": self.px_to_mm(radius),
            "diameter_mm": self.px_to_mm(radius * 2),
            "contour_points": box.tolist(),
            "center": (int(cx), int(cy)),
        }

    def _empty_result(self, label, bbox):
        return {
            "label": label, "bbox": bbox,
            "width_px": 0, "height_px": 0,
            "width_mm": None, "height_mm": None,
            "area_px": 0, "area_mm2": None,
            "perimeter_px": 0, "perimeter_mm": None,
            "circularity": 0, "solidity": 0,
            "aspect_ratio": 0, "radius_px": 0, "radius_mm": None,
            "diameter_mm": None, "contour_points": [], "center": (0, 0),
        }

    def measure_object(self, bbox, label=""):
        """兼容旧接口的快速测量，不做亚像素"""
        x1, y1, x2, y2 = bbox
        w_px = x2 - x1
        h_px = y2 - y1

        return {
            "label": label, "bbox": [x1, y1, x2, y2],
            "width_px": int(w_px), "height_px": int(h_px),
            "width_mm": self.px_to_mm(w_px), "height_mm": self.px_to_mm(h_px),
            "area_px": int(w_px * h_px),
            "area_mm2": round(self.px_to_mm(w_px * h_px), 2) if self.pixel_per_mm else None,
            "perimeter_px": 2 * int(w_px + h_px),
            "perimeter_mm": self.px_to_mm(2 * (w_px + h_px)),
            "circularity": 0, "solidity": 0, "aspect_ratio": round(max(w_px, h_px) / (min(w_px, h_px) + 1e-6), 3),
            "radius_px": int(max(w_px, h_px) / 2),
            "radius_mm": self.px_to_mm(max(w_px, h_px) / 2),
            "diameter_mm": self.px_to_mm(max(w_px, h_px)),
            "contour_points": [],
            "center": (int(x1 + w_px / 2), int(y1 + h_px / 2)),
        }

    def find_reference_object(self, frame: np.ndarray) -> Tuple[Optional[float], np.ndarray]:
        """
        自动寻找参考物（增强版）
        支持：Hough圆检测 + 矩形轮廓检测 + 颜色分割
        """
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # ---- 方法1: Hough 圆检测 ----
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1.2, 100,
            param1=self.params.circle_param1,
            param2=self.params.circle_param2,
            minRadius=self.params.circle_min_r,
            maxRadius=self.params.circle_max_r,
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            # 取最大圆作为参考物
            best = max(circles, key=lambda c: c[2])
            x, y, r = best
            ref_width_px = 2 * r
            self.set_pixel_per_mm(ref_width_px)

            # 精美标注
            cv2.circle(annotated, (x, y), r, (0, 255, 100), 2)
            cv2.circle(annotated, (x, y), 3, (0, 255, 100), -1)

            # 直径线
            cv2.line(annotated, (x - r, y), (x + r, y), (0, 255, 100), 2)
            cv2.line(annotated, (x - r, y - 5), (x - r, y + 5), (0, 255, 100), 2)
            cv2.line(annotated, (x + r, y - 5), (x + r, y + 5), (0, 255, 100), 2)

            label = f"REF: D={ref_width_px}px ≈ {self.ref_width_mm}{REAL_UNIT}"
            cv2.putText(annotated, label, (x - r, y - r - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
            return ref_width_px, annotated

        # ---- 方法2: 最大矩形轮廓 ----
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 过滤：面积不能太大（大于画面80%的不是参考物）
            img_area = frame.shape[0] * frame.shape[1]
            valid_cnts = [c for c in contours
                         if 500 < cv2.contourArea(c) < 0.8 * img_area]

            if valid_cnts:
                largest = max(valid_cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                w, h = rect[1]
                ref_width_px = max(w, h)

                # 过滤太细长的
                if min(w, h) / max(w, h) > 0.2:
                    self.set_pixel_per_mm(ref_width_px)
                    cv2.drawContours(annotated, [box], 0, (0, 255, 100), 2)

                    # 正交尺寸标注
                    pts = box.tolist()
                    text_pt = (min(p[0] for p in pts), min(p[1] for p in pts) - 10)
                    label = f"REF: {ref_width_px:.0f}px ≈ {self.ref_width_mm}{REAL_UNIT}"
                    cv2.putText(annotated, label, text_pt,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
                    return ref_width_px, annotated

        return None, annotated

    def draw_measurement(self, frame, measurements):
        """
        绘制测量结果（增强版：带尺寸线与几何标注）
        """
        result = frame.copy()

        for i, m in enumerate(measurements):
            color = (255, 80, 80) if i == 0 else (255, 165, 0)  # 主对象红色，其余橙色

            # ---- 绘制精确轮廓 ----
            pts = m.get("contour_points", [])
            if pts and len(pts) >= 4:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(result, [pts_arr], True, color, 2)

            # ---- 绘制边界框（虚线效果） ----
            x1, y1, x2, y2 = m["bbox"]
            for k in range(0, max(x2 - x1, y2 - y1), 10):
                cv2.line(result, (x1 + k, y1), (min(x1 + k + 5, x2), y1), color, 1)
                cv2.line(result, (x1 + k, y2), (min(x1 + k + 5, x2), y2), color, 1)
                cv2.line(result, (x1, y1 + k), (x1, min(y1 + k + 5, y2)), color, 1)
                cv2.line(result, (x2, y1 + k), (x2, min(y1 + k + 5, y2)), color, 1)

            # ---- 同心十字中心 ----
            cx, cy = m.get("center", (0, 0))
            cross = 8
            if cx > 0 and cy > 0:
                cv2.line(result, (cx - cross, cy), (cx + cross, cy), (0, 255, 255), 1)
                cv2.line(result, (cx, cy - cross), (cx, cy + cross), (0, 255, 255), 1)

            # ---- 正交尺寸线 ----
            if self.params.show_measure_lines:
                # 宽度线（底边）
                mx = int((x1 + x2) / 2)
                line_y = y2 + 20
                cv2.line(result, (x1, line_y), (x2, line_y), color, 2)
                cv2.line(result, (x1, line_y - 5), (x1, line_y + 5), color, 2)
                cv2.line(result, (x2, line_y - 5), (x2, line_y + 5), color, 2)

                # 高度线（右边）
                line_x = x2 + 20
                cv2.line(result, (line_x, y1), (line_x, y2), color, 2)
                cv2.line(result, (line_x - 5, y1), (line_x + 5, y1), color, 2)
                cv2.line(result, (line_x - 5, y2), (line_x + 5, y2), color, 2)

                # 尺寸文字
                w_mm = m.get("width_mm")
                h_mm = m.get("height_mm")
                if w_mm:
                    w_label = f"W:{w_mm}{REAL_UNIT}"
                    cv2.putText(result, w_label, (mx - 30, line_y + 18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if h_mm:
                    h_label = f"H:{h_mm}{REAL_UNIT}"
                    cv2.putText(result, h_label, (line_x + 6, int((y1 + y2) / 2)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ---- 标签（含几何特征） ----
            label_parts = [m.get("label", "object")]
            if m.get("width_mm") and m.get("height_mm"):
                label_parts.append(f"{m['width_mm']}×{m['height_mm']}{REAL_UNIT}")
            if m.get("area_mm2"):
                label_parts.append(f"S≈{m['area_mm2']}{REAL_UNIT}²")
            if m.get("diameter_mm"):
                label_parts.append(f"D≈{m['diameter_mm']}{REAL_UNIT}")

            label = " | ".join(label_parts)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            label_bg_y = y1 - th - 8
            if label_bg_y < 0:
                label_bg_y = y2 + 40

            cv2.rectangle(result, (x1, label_bg_y), (x1 + tw + 6, label_bg_y + th + baseline + 4),
                         color, -1)
            cv2.putText(result, label, (x1 + 3, label_bg_y + th + baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return result

    def calibrate_camera(self, checkerboard_images, checkerboard_size=(9, 6)):
        """相机标定（使用棋盘格）"""
        print("[Dimension] 开始相机标定...")
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

        objpoints, imgpoints = [], []

        for img_path in checkerboard_images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if not objpoints:
            raise RuntimeError("未找到足够的棋盘格角点，请检查图像")

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.is_calibrated = True
        print(f"[Dimension] 标定完成! camera_matrix:\n{camera_matrix}")
        return camera_matrix, dist_coeffs, rvecs, tvecs


# ============ 便捷函数 ============

def quick_measure(frame, bbox_list):
    """快速测量"""
    measurer = DimensionMeasurer()
    ref_width_px, frame_with_ref = measurer.find_reference_object(frame)

    if ref_width_px is None:
        measurements = [measurer.measure_object(bbox, f"obj_{i}")
                       for i, bbox in enumerate(bbox_list)]
        result = frame.copy()
        for m in measurements:
            x1, y1, x2, y2 = m["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result, f"{m['width_px']}×{m['height_px']}px",
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(result, "Place reference object (coin/A4) for real mm",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return result, measurements

    measurements = [measurer.measure_object(bbox, f"obj_{i}")
                   for i, bbox in enumerate(bbox_list)]
    result = measurer.draw_measurement(frame_with_ref, measurements)
    return result, measurements
