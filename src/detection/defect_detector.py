"""
缺陷检测模块 - 基于 YOLOv11
支持：预训练模型推理、自定义数据集训练、实时视频流检测
"""

import cv2
import numpy as np
from ultralytics import YOLO
from ..utils.config import CONF_THRESHOLD, IOU_THRESHOLD, YOLO_MODEL_NAME, TRAINED_MODEL_PATH, MODELS_DIR, DEFECT_CLASSES


class DefectDetector:
    """
    工业缺陷检测器
    封装 YOLO 模型加载、推理、结果可视化
    """

    def __init__(self, model_path=None, conf=None, iou=None):
        """
        Args:
            model_path: 模型路径，None时自动选择（优先使用训练好的模型）
            conf: 置信度阈值，None时使用配置文件值
            iou:  IOU阈值，None时使用配置文件值
        """
        self.conf = conf or CONF_THRESHOLD
        self.iou = iou or IOU_THRESHOLD
        self.class_names = DEFECT_CLASSES

        # 自动选择模型：优先使用训练好的模型
        if model_path is None:
            if TRAINED_MODEL_PATH is not None:
                model_path = TRAINED_MODEL_PATH
                print(f"[DefectDetector] 使用训练好的模型: {model_path}")
            else:
                model_path = YOLO_MODEL_NAME
                print(f"[DefectDetector] 使用预训练模型: {model_path}")

        # 加载模型
        self.model = YOLO(model_path)
        print(f"[DefectDetector] 模型加载成功")

        # 获取模型实际类别（如果是以训练模型）
        if hasattr(self.model, 'names') and self.model.names:
            self.class_names = list(self.model.names.values())

    def predict(self, frame):
        """
        对单帧进行缺陷检测
        Args:
            frame: OpenCV BGR 图像 (numpy array)
        Returns:
            results: YOLO 推理结果列表
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )
        return results

    def draw_results(self, frame, results):
        """
        在帧上绘制检测结果
        Args:
            frame:   原始图像
            results: predict() 的返回结果
        Returns:
            annotated_frame: 标注后的图像
            detections: 检测结果列表 [{"class":, "confidence":, "bbox":}]
        """
        annotated_frame = frame.copy()
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                # 解析检测框
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

                # 序号
                idx = len(detections) + 1

                # 记录检测结果
                detections.append({
                    "id": idx,
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "width": x2 - x1,
                    "height": y2 - y1,
                })

                # 绘制边界框（不同类别用不同颜色）
                color = self._get_class_color(cls_id)

                # 左上角：大号序号角标
                badge_size = 30
                cv2.rectangle(annotated_frame, (x1, y1), (x1 + badge_size, y1 + badge_size), color, -1)
                cv2.putText(annotated_frame, str(idx),
                           (x1 + 7, y1 + badge_size - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # 绘制边界框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"#{idx} {cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame, detections

    def process_frame(self, frame):
        """
        一站式处理：推理 + 绘图
        Returns:
            annotated_frame: 标注后的图像
            detections:      检测结果列表
        """
        results = self.predict(frame)
        annotated_frame, detections = self.draw_results(frame, results)
        return annotated_frame, detections

    @staticmethod
    def _get_class_color(cls_id):
        """不同类别分配不同颜色（BGR）"""
        palette = [
            (0, 255, 0),    # 亮绿
            (255, 80, 80),  # 红
            (255, 200, 0),  # 金橙
            (0, 180, 255),  # 橙
            (200, 0, 255),  # 紫
            (255, 0, 160),  # 品红
            (0, 255, 255),  # 青
            (80, 180, 80),  # 橄榄绿
            (255, 255, 0),  # 蓝绿
            (0, 120, 255),  # 深橙
        ]
        return palette[cls_id % len(palette)]

    def train(self, data_yaml_path, epochs=100, img_size=640, batch_size=16):
        """
        训练自定义缺陷检测模型
        Args:
            data_yaml_path: 数据集配置文件路径 (data.yaml)
            epochs:         训练轮数
            img_size:       输入图像尺寸
            batch_size:     批次大小
        """
        print(f"[DefectDetector] 开始训练，数据集: {data_yaml_path}")
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            project=MODELS_DIR,
            name="defect_detection",
            exist_ok=True,
        )
        return results

    def export(self, format="onnx"):
        """导出模型（用于部署）"""
        return self.model.export(format=format)


# ============ 简易演示：传统CV缺陷检测（无需训练，开箱即用）============

def detect_defects_traditional(frame, method="edge"):
    """
    使用传统计算机视觉方法检测缺陷（无需训练数据）
    适合作为 demo 展示和传统方法对比

    Args:
        frame:  输入图像
        method: "edge"(边缘检测) / "threshold"(阈值) / "contour"(轮廓分析)
    Returns:
        result_frame: 标注后的图像
        defect_regions: 检测到的缺陷区域列表
    """
    result = frame.copy()
    defect_regions = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if method == "edge":
        # Canny 边缘检测 + 形态学处理
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000:  # 过滤太小/太大的区域
                x, y, w, h = cv2.boundingRect(cnt)
                defect_regions.append({"bbox": [x, y, x+w, y+h], "area": area, "method": "edge"})
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(result, f"defect({int(area)})", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    elif method == "threshold":
        # 自适应阈值 + 轮廓检测
        thresh = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                defect_regions.append({"bbox": [x, y, x+w, y+h], "area": area, "method": "threshold"})
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 165, 255), 2)

    return result, defect_regions
