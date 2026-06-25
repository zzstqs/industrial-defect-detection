"""
传统计算机视觉缺陷检测模块 - 多通道融合检测
支持：预处理管线 + Canny边缘 / 自适应阈值 / LBP纹理 / FFT频域 / HSV颜色空间
每个通道独立运行，结果融合排序，适合展示经典 CV 算法能力
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ============ 参数配置 ============

@dataclass
class TraditionalCVParams:
    """传统CV所有可调参数"""

    # --- 预处理 ---
    denoise_kernel: int = 5           # 高斯/中值滤波核大小 (奇数)
    denoise_method: str = "gaussian"  # "gaussian" | "median" | "bilateral"
    use_clahe: bool = True            # 是否启用 CLAHE 直方图均衡
    clahe_clip: float = 2.0          # CLAHE 对比度限制
    sharpen_strength: float = 0.0    # 锐化强度 (0=关闭, 0.3~1.5)

    # --- Canny 边缘检测 ---
    enable_canny: bool = True
    canny_low: int = 50
    canny_high: int = 150
    morph_kernel: int = 3            # 形态学操作核大小
    area_min: int = 500              # 最小轮廓面积（提高过滤噪声）
    area_max: int = 50000            # 最大轮廓面积

    # --- 自适应阈值 ---
    enable_threshold: bool = True
    thresh_block: int = 11           # 自适应阈值块大小 (奇数)
    thresh_c: int = 2               # 自适应阈值常数
    thresh_area_min: int = 5000

    # --- LBP 纹理分析（默认关闭，网格模式框太多） ---
    enable_lbp: bool = False
    lbp_radius: int = 1
    lbp_points: int = 8
    lbp_grid_rows: int = 8          # 将图像分成 grid_rows × grid_cols 块
    lbp_grid_cols: int = 8
    lbp_anomaly_std: float = 2.5    # 异常阈值 (标准差倍数)

    # --- FFT 频域分析 ---
    enable_fft: bool = True
    fft_highpass_radius: int = 30   # 高通滤波半径 (越大保留越多低频)

    # --- HSV 颜色空间 ---
    enable_hsv: bool = True
    hsv_low: Tuple[int, int, int] = (0, 20, 50)     # HSV 下限 (针对暗色缺陷)
    hsv_high: Tuple[int, int, int] = (180, 255, 200) # HSV 上限
    hsv_inv_low: Tuple[int, int, int] = (0, 0, 200)  # 亮色缺陷 HSV 下限
    hsv_inv_high: Tuple[int, int, int] = (180, 50, 255) # 亮色缺陷 HSV 上限

    # --- 结果过滤与融合 ---
    confidence_threshold: float = 0.4  # 最低置信度（0=不过滤, 推荐0.3~0.6）
    iou_threshold: float = 0.15        # 多通道结果融合 IOU 阈值（越低越严格）
    max_detections: int = 50           # 最大输出检测框数量

    # --- 颜色 ---
    class_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "edge": (0, 255, 255),       # 青
        "threshold": (255, 165, 0),  # 橙
        "lbp": (255, 0, 255),        # 品红
        "fft": (0, 255, 0),          # 绿
        "hsv": (255, 80, 80),        # 红
        "merged": (0, 200, 255),     # 金
    })


class PreprocessMethod(Enum):
    NONE = "none"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"


# ============ 预处理管线 ============

class PreprocessingPipeline:
    """完整的传统CV预处理管线"""

    def __init__(self, params: TraditionalCVParams):
        self.params = params

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        预处理流程：去噪 → 灰度化 → 均衡化 → 锐化
        Returns:
            processed: 最终处理后的灰度图
            color: 处理后的彩色图
            intermediates: 中间结果 dict (用于调试/展示)
        """
        intermediates = {}
        color = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intermediates["raw_gray"] = gray.copy()
        intermediates["raw_color"] = color.copy()

        # Step 1: 去噪
        ksize = self.params.denoise_kernel
        if ksize % 2 == 0:
            ksize += 1

        if self.params.denoise_method == "gaussian":
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            color = cv2.GaussianBlur(color, (ksize, ksize), 0)
        elif self.params.denoise_method == "median":
            gray = cv2.medianBlur(gray, ksize)
            color = cv2.medianBlur(color, ksize)
        elif self.params.denoise_method == "bilateral":
            gray = cv2.bilateralFilter(gray, ksize, 75, 75)
            color = cv2.bilateralFilter(color, ksize, 75, 75)

        intermediates["denoised_gray"] = gray.copy()

        # Step 2: CLAHE 直方图均衡化
        if self.params.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.params.clahe_clip, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            intermediates["clahe"] = gray.copy()

        # Step 3: 锐化 (Unsharp Mask)
        if self.params.sharpen_strength > 0.001:
            blurred = cv2.GaussianBlur(gray, (0, 0), 3)
            gray = cv2.addWeighted(gray, 1.0 + self.params.sharpen_strength,
                                    blurred, -self.params.sharpen_strength, 0)
            intermediates["sharpened"] = gray.copy()

        return gray, color, intermediates


# ============ 多通道检测器 ============

class CannyEdgeDetector:
    """Canny 边缘检测 + 形态学处理"""

    def detect(self, gray: np.ndarray, params: TraditionalCVParams) -> List[Dict]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, params.canny_low, params.canny_high)

        # 形态学闭运算连接断裂边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (params.morph_kernel, params.morph_kernel))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if params.area_min < area < params.area_max:
                x, y, w, h = cv2.boundingRect(cnt)
                # 计算更多特征
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / (hull_area + 1e-6)

                results.append({
                    "bbox": [x, y, x + w, y + h],
                    "area": int(area),
                    "perimeter": int(perimeter),
                    "circularity": round(circularity, 3),
                    "solidity": round(solidity, 3),
                    "method": "edge",
                    "confidence": min(1.0, circularity * 0.7 + solidity * 0.3),
                    "width": w,
                    "height": h,
                })

        return results


class AdaptiveThresholdDetector:
    """自适应阈值检测"""

    def detect(self, gray: np.ndarray, params: TraditionalCVParams) -> List[Dict]:
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, params.thresh_block, params.thresh_c
        )

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > params.thresh_area_min:
                x, y, w, h = cv2.boundingRect(cnt)
                # 排除太细长的（可能是噪声线）
                aspect_ratio = max(w, h) / (min(w, h) + 1)
                if aspect_ratio > 15 and area < 500:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

                results.append({
                    "bbox": [x, y, x + w, y + h],
                    "area": int(area),
                    "perimeter": int(perimeter),
                    "circularity": round(circularity, 3),
                    "solidity": 1.0,
                    "method": "threshold",
                    "confidence": min(0.85, 0.3 + area / 2000),
                    "width": w,
                    "height": h,
                })

        return results


class LBPTextureAnalyzer:
    """LBP 纹理异常检测"""

    def _compute_lbp(self, gray: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """计算 LBP 特征图"""
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                pattern = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(j + radius * np.cos(angle))
                    y = int(i - radius * np.sin(angle))
                    if 0 <= x < w and 0 <= y < h:
                        pattern |= (gray[y, x] >= center) << k
                lbp[i, j] = pattern % 256
        return lbp

    def detect(self, gray: np.ndarray, params: TraditionalCVParams) -> List[Dict]:
        h, w = gray.shape
        grid_h = h // params.lbp_grid_rows
        grid_w = w // params.lbp_grid_cols

        if grid_h < 8 or grid_w < 8:
            return []  # 图像太小

        # 计算 LBP 特征图
        lbp_img = self._compute_lbp(gray, params.lbp_radius, params.lbp_points)

        # 计算每块的 LBP 直方图
        histograms = []
        positions = []
        for i in range(params.lbp_grid_rows):
            for j in range(params.lbp_grid_cols):
                y1, y2 = i * grid_h, (i + 1) * grid_h
                x1, x2 = j * grid_w, (j + 1) * grid_w
                block = lbp_img[y1:y2, x1:x2]
                hist = cv2.calcHist([block], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                histograms.append(hist)
                positions.append((x1, y1, x2, y2))

        if len(histograms) < 2:
            return []

        # 计算全局平均直方图，找出异常块
        histograms = np.array(histograms)
        mean_hist = histograms.mean(axis=0)
        std_hist = histograms.std(axis=0) + 1e-6

        results = []
        seen_rects = set()

        for idx, (hist, (x1, y1, x2, y2)) in enumerate(zip(histograms, positions)):
            # 卡方距离
            dist = np.sum((hist - mean_hist) ** 2 / (mean_hist + 1e-6))
            z_score = (dist - np.mean([np.sum((h - mean_hist) ** 2 / (mean_hist + 1e-6))
                                        for h in histograms])) / 1e-6

            if dist > np.mean([np.sum((h - mean_hist) ** 2 / (mean_hist + 1e-6))
                               for h in histograms]) * params.lbp_anomaly_std:
                key = (x1 // 10, y1 // 10)
                if key not in seen_rects:
                    seen_rects.add(key)
                    results.append({
                        "bbox": [x1, y1, x2, y2],
                        "area": (x2 - x1) * (y2 - y1),
                        "method": "lbp",
                        "confidence": min(0.7, dist / 100),
                        "width": x2 - x1,
                        "height": y2 - y1,
                    })

        return results


class FFTAnomalyDetector:
    """FFT 频域异常检测 - 检测高频缺陷（裂纹、划痕）"""

    def detect(self, gray: np.ndarray, params: TraditionalCVParams) -> List[Dict]:
        h, w = gray.shape

        # FFT 变换
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)

        # 高通滤波：只保留高频成分（缺陷通常表现为高频）
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), params.fft_highpass_radius, 0, -1)

        fshift_filtered = fshift * mask

        # 逆变换
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # 归一化
        if img_back.max() > 0:
            img_back = (img_back / img_back.max() * 255).astype(np.uint8)

        # 二值化
        _, binary = cv2.threshold(img_back, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if params.area_min < area < params.area_max:
                x, y, w, h = cv2.boundingRect(cnt)
                results.append({
                    "bbox": [x, y, x + w, y + h],
                    "area": int(area),
                    "method": "fft",
                    "confidence": min(0.8, 0.3 + area / 5000),
                    "width": w,
                    "height": h,
                })

        return results


class HSVColorDetector:
    """HSV 颜色空间缺陷检测"""

    def detect(self, frame: np.ndarray, params: TraditionalCVParams) -> List[Dict]:
        """检测颜色异常的缺陷（锈斑、变色、污渍等）"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 通道1：暗色缺陷
        mask_dark = cv2.inRange(hsv, np.array(params.hsv_low), np.array(params.hsv_high))
        # 通道2：亮色缺陷
        mask_light = cv2.inRange(hsv, np.array(params.hsv_inv_low), np.array(params.hsv_inv_high))
        mask = cv2.bitwise_or(mask_dark, mask_light)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if params.area_min < area < params.area_max:
                x, y, w, h = cv2.boundingRect(cnt)

                # 计算区域内的颜色方差（颜色异常程度）
                roi = hsv[y:y + h, x:x + w]
                if roi.size > 0:
                    h_std = np.std(roi[:, :, 0])
                    s_std = np.std(roi[:, :, 1])
                    color_anomaly = min(1.0, (h_std + s_std) / 100)
                else:
                    color_anomaly = 0.5

                results.append({
                    "bbox": [x, y, x + w, y + h],
                    "area": int(area),
                    "method": "hsv",
                    "confidence": min(0.85, color_anomaly * 0.9 + area / 3000),
                    "width": w,
                    "height": h,
                })

        return results


# ============ 传统CV主类 ============

class TraditionalCVDetector:
    """传统计算机视觉缺陷检测主控类"""

    def __init__(self, params: Optional[TraditionalCVParams] = None):
        self.params = params or TraditionalCVParams()
        self.preprocessor = PreprocessingPipeline(self.params)
        self.edge_detector = CannyEdgeDetector()
        self.thresh_detector = AdaptiveThresholdDetector()
        self.lbp_analyzer = LBPTextureAnalyzer()
        self.fft_detector = FFTAnomalyDetector()
        self.hsv_detector = HSVColorDetector()

        # 可视化中间结果
        self.intermediates: Dict[str, np.ndarray] = {}
        self.channel_results: Dict[str, List[Dict]] = {}

    def set_params(self, **kwargs):
        """实时更新参数"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        多通道融合检测
        Returns:
            annotated_frame: 标注后的图像
            merged_results:  融合排序后的检测结果
        """
        result_frame = frame.copy()
        all_results = []

        # 1. 预处理
        gray, color, intermediates = self.preprocessor.process(frame)
        self.intermediates = intermediates

        # 2. 各通道检测
        self.channel_results = {}

        if self.params.enable_canny:
            edge_results = self.edge_detector.detect(gray, self.params)
            self.channel_results["edge"] = edge_results
            all_results.extend(edge_results)

        if self.params.enable_threshold:
            thresh_results = self.thresh_detector.detect(gray, self.params)
            self.channel_results["threshold"] = thresh_results
            all_results.extend(thresh_results)

        if self.params.enable_lbp:
            lbp_results = self.lbp_analyzer.detect(gray, self.params)
            self.channel_results["lbp"] = lbp_results
            all_results.extend(lbp_results)

        if self.params.enable_fft:
            fft_results = self.fft_detector.detect(gray, self.params)
            self.channel_results["fft"] = fft_results
            all_results.extend(fft_results)

        if self.params.enable_hsv:
            hsv_results = self.hsv_detector.detect(frame, self.params)
            self.channel_results["hsv"] = hsv_results
            all_results.extend(hsv_results)

        # 3. 置信度过滤（先过滤，再 NMS，大幅减少 NMS 计算量）
        conf_thresh = self.params.confidence_threshold
        if conf_thresh > 0:
            all_results = [r for r in all_results if r.get("confidence", 0) >= conf_thresh]

        # 4. NMS 融合（去重）
        merged = self._nms_fusion(all_results)

        # 5. 限制最大数量
        if len(merged) > self.params.max_detections:
            merged = merged[:self.params.max_detections]

        # 6. 绘制结果
        result_frame = self._draw_results(result_frame, merged)

        return result_frame, merged

    def _nms_fusion(self, results: List[Dict]) -> List[Dict]:
        """基于 IOU 的 NMS 融合，保留置信度最高者"""
        if not results:
            return []

        # 按置信度降序
        results = sorted(results, key=lambda r: r.get("confidence", 0), reverse=True)

        boxes = np.array([r["bbox"] for r in results], dtype=np.float32)
        scores = np.array([r.get("confidence", 0.5) for r in results], dtype=np.float32)

        keep_indices = []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue
            keep_indices.append(i)
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                iou = self._iou(boxes[i], boxes[j])
                if iou > self.params.iou_threshold:
                    used.add(j)

        merged = [results[i] for i in keep_indices]
        # 按面积降序（大缺陷排在前面更显眼）
        merged = sorted(merged, key=lambda r: r.get("area", 0), reverse=True)

        return merged

    @staticmethod
    def _iou(box_a, box_b):
        """计算两个边界框的 IOU"""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制多通道检测结果"""
        annotated = frame.copy()

        for idx, r in enumerate(results):
            x1, y1, x2, y2 = r["bbox"]
            method = r.get("method", "merged")
            conf = r.get("confidence", 0.5)

            # 颜色：根据检测方法
            color = self.params.class_colors.get(method, (0, 200, 255))

            # 置信度映射到颜色亮度
            color = tuple(int(c * (0.5 + 0.5 * conf)) for c in color)

            # 序号角标
            badge_size = 28
            cv2.rectangle(annotated, (x1, y1), (x1 + badge_size, y1 + badge_size), color, -1)
            cv2.putText(annotated, str(idx + 1),
                       (x1 + 7, y1 + badge_size - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # 边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 标签：序号 + 方法 + 置信度
            label = f"#{idx + 1} {method} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = y1 - th - 8
            if label_y < 0:
                label_y = y1 + badge_size + 4  # 框上方空间不够，画在角标下面

            cv2.rectangle(annotated, (x1, label_y), (x1 + tw + 4, label_y + th + baseline + 4), color, -1)
            cv2.putText(annotated, label, (x1 + 2, label_y + th + baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 如果面积很大，画描边突出显示
            if r.get("area", 0) > 5000:
                cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

        return annotated

    def get_channel_summary(self) -> Dict:
        """获取各通道检测概况（用于前端展示）"""
        return {
            "channels": {
                name: len(results)
                for name, results in self.channel_results.items()
            },
            "total_raw": sum(len(r) for r in self.channel_results.values()),
        }


# ============ 兼容旧接口 ============

def detect_defects_traditional(frame, method="all", params: Optional[TraditionalCVParams] = None):
    """
    传统CV缺陷检测（兼容旧接口）
    Args:
        frame:  输入图像
        method: "all" | "edge" | "threshold" | "lbp" | "fft" | "hsv"
        params: 参数配置，None 时使用默认参数
    Returns:
        result_frame: 标注后的图像
        defect_regions: 检测结果列表
    """
    detector = TraditionalCVDetector(params)

    if method == "edge":
        gray, _, _ = detector.preprocessor.process(frame)
        results = detector.edge_detector.detect(gray, detector.params)
        frame = detector._draw_results(frame, results)
    elif method == "threshold":
        gray, _, _ = detector.preprocessor.process(frame)
        results = detector.thresh_detector.detect(gray, detector.params)
        frame = detector._draw_results(frame, results)
    else:
        # "all" - 全通道融合
        frame, results = detector.detect(frame)

    return frame, results
