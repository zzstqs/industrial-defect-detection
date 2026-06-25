"""
WebSocket 视频流处理器
负责：摄像头帧采集、检测/测量编排、结果稳定化、帧编码与推送

v2.1: 将同步阻塞操作（YOLO推理、CV检测、JPEG编码）通过 asyncio.to_thread()
      卸载到线程池执行，确保 asyncio 事件循环始终可响应 WebSocket ping/pong，
      彻底解决"开检测就断连"的问题。
"""

import cv2
import base64
import asyncio
import time
import json
import numpy as np
from typing import Optional, List, Dict

from fastapi import WebSocket, WebSocketDisconnect

from ..state import get_state, update
from ...utils.camera import CameraCapture, frame_to_jpeg_bytes
from ..detection_stabilizer import DetectionStabilizer


# 全局摄像头实例（单例，因为物理摄像头只有一个）
_camera: Optional[CameraCapture] = None
_stabilizer: Optional[DetectionStabilizer] = None


def _get_camera() -> CameraCapture:
    """获取或创建摄像头实例"""
    global _camera
    if _camera is None:
        state = get_state()
        _camera = CameraCapture(source=state.camera_source).start()
        update(camera_active=True)
        print(f"[VideoHandler] 摄像头已启动 (source={state.camera_source})")
    return _camera


def _get_stabilizer() -> DetectionStabilizer:
    """获取或创建检测结果稳定器"""
    global _stabilizer
    if _stabilizer is None:
        _stabilizer = DetectionStabilizer(
            sticky_frames=8,    # 结果消失后保持 8 帧 (~267ms @30fps)
            confirm_frames=3,   # 新结果需持续 3 帧 (~100ms) 才确认显示
        )
        update(stabilizer=_stabilizer)
        print("[VideoHandler] 检测稳定器已初始化")
    return _stabilizer


# 帧处理耗时告警阈值（毫秒）
SLOW_FRAME_WARN_MS = 200


class VideoStreamHandler:
    """
    WebSocket 视频流处理器

    v2.1 架构：所有阻塞操作通过 asyncio.to_thread() 在线程池中执行
    ├── _process_single_frame() — 同步方法，在线程池线程中运行
    │   ├── 摄像头读取
    │   ├── 图像缩放
    │   ├── 缺陷检测 (YOLO GPU推理 / 传统CV)
    │   ├── 检测结果稳定化
    │   ├── 尺寸测量
    │   └── JPEG编码 + base64
    └── run() — 异步主循环
        ├── await to_thread(_process_single_frame)  ← 关键：不阻塞事件循环
        ├── await websocket.send_text()              ← 唯一的异步操作
        └── await asyncio.sleep()                     ← 帧率控制
    """

    def __init__(
        self,
        detection_service=None,
        measurement_service=None,
        websocket: Optional[WebSocket] = None,
    ):
        self.detection_service = detection_service
        self.measurement_service = measurement_service
        self.ws = websocket

        # 运行统计
        self.frame_idx = 0
        self.last_log_time = time.time()
        self.log_interval = 30  # 每30帧输出一次性能日志

        # 参考物跟踪
        self.ref_detected = False
        self.ref_last_seen = 0

    # ============ 同步帧处理（运行在线程池中）============

    def _process_single_frame(
        self, camera: CameraCapture, stabilizer: DetectionStabilizer, timestamp: float
    ) -> Optional[Dict]:
        """
        单帧完整处理管线（同步方法，运行在线程池线程中）

        返回: payload dict，失败返回 None
        """
        # 1. 读取摄像头帧
        ret, frame = camera.read()
        if not ret:
            return None

        # 2. 缩放到 640 宽以加速处理
        h_orig, w_orig = frame.shape[:2]
        if w_orig > 640:
            scale = 640.0 / w_orig
            frame = cv2.resize(frame, (640, int(h_orig * scale)),
                               interpolation=cv2.INTER_AREA)

        # 3. 缺陷检测（YOLO GPU推理 或 传统CV）
        raw_detections = []
        if self.detection_service:
            try:
                processed_frame, raw_detections = (
                    self.detection_service.process_and_annotate(frame)
                )
            except Exception as e:
                print(f"[VideoHandler] 检测服务异常: {e}")
                processed_frame = frame.copy()
                raw_detections = []
        else:
            processed_frame = frame.copy()

        # 4. 检测结果稳定化
        stable_detections = stabilizer.feed(raw_detections)

        # 5. 尺寸测量（始终执行：无检测时自动找参考物）
        measurement_results = []
        if self.measurement_service:
            try:
                processed_frame, measurement_results, self.ref_detected = (
                    self.measurement_service.measure(processed_frame, stable_detections)
                )
            except Exception as e:
                print(f"[VideoHandler] 测量服务异常: {e}")
                measurement_results = []

        # 6. 编码为 JPEG -> base64
        jpeg_bytes = frame_to_jpeg_bytes(processed_frame, quality=75)
        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

        # 7. 构建推送载荷（不含 numpy 类型的纯 Python dict）
        state = get_state()
        status_info = {}
        if state.detection_enabled and not raw_detections:
            status_info["detection_status"] = "no_defect"
        if state.measurement_enabled and not measurement_results and self.ref_detected:
            status_info["measurement_status"] = "reference_only"

        return {
            "frame": b64,
            "timestamp": timestamp,
            "detections": stable_detections,
            "measurements": measurement_results,
            "status": status_info,
            "is_stable": True,
        }

    # ============ 异步主循环 ============

    async def run(self, websocket: WebSocket):
        """异步主循环 — 帧处理卸载到线程池，事件循环始终响应"""
        self.ws = websocket
        await websocket.accept()
        print("[VideoHandler] WebSocket 客户端已连接")

        camera = _get_camera()
        stabilizer = _get_stabilizer()
        consecutive_errors = 0
        max_consecutive_errors = 10

        try:
            while True:
                t0 = time.time()

                try:
                    # === 关键：所有阻塞操作通过 to_thread 卸载到线程池 ===
                    # 这样 asyncio 事件循环可以继续处理 WebSocket ping/pong
                    # 以及其他协程（API 请求等）
                    payload = await asyncio.to_thread(
                        self._process_single_frame, camera, stabilizer, t0
                    )

                    if payload is None:
                        # 摄像头读取失败（非致命，跳过这帧）
                        await asyncio.sleep(0.01)
                        continue

                except WebSocketDisconnect:
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    print(f"[VideoHandler] 帧#{self.frame_idx} 线程池异常: {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[VideoHandler] 连续 {consecutive_errors} 帧失败，中止流")
                        raise
                    await asyncio.sleep(0.1)
                    continue

                # === 仅 WebSocket 发送在异步上下文中 ===
                try:
                    await websocket.send_text(
                        json.dumps(payload, default=_json_serializer)
                    )
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    print(f"[VideoHandler] 帧#{self.frame_idx} 发送失败: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise
                    await asyncio.sleep(0.1)
                    continue

                # 成功处理 → 重置错误计数
                consecutive_errors = 0
                self.frame_idx += 1

                # 8. 性能日志
                t_total = (time.time() - t0) * 1000
                self._log_performance(t0, t_total)

                # 9. 帧率控制
                sleep_ms = max(5, 33 - t_total)
                await asyncio.sleep(sleep_ms / 1000.0)

        except WebSocketDisconnect:
            print("[VideoHandler] 客户端断开连接")
        except asyncio.CancelledError:
            print("[VideoHandler] 任务被取消")
        except Exception as e:
            print(f"[VideoHandler] 致命错误，流中止: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[VideoHandler] 会话结束，共处理 {self.frame_idx} 帧")

    def _log_performance(self, t_start: float, elapsed_ms: float):
        """每N帧输出性能日志，慢帧告警"""
        # 慢帧告警（不依赖 log_interval）
        if elapsed_ms > SLOW_FRAME_WARN_MS:
            print(
                f"[VideoHandler] ⚠️ 慢帧告警 帧#{self.frame_idx}: "
                f"耗时 {elapsed_ms:.0f}ms (阈值 {SLOW_FRAME_WARN_MS}ms)"
            )

        if self.frame_idx % self.log_interval != 0:
            return

        interval_elapsed = (time.time() - self.last_log_time) * 1000
        fps = self.log_interval / (interval_elapsed / 1000) if interval_elapsed > 0 else 0

        state = get_state()
        stabilizer_stats = {}
        if state.stabilizer:
            stabilizer_stats = state.stabilizer.get_stats()

        method_label = state.detection_method
        if state.detection_method == "yolo" and state.detector is not None:
            has_model = hasattr(state.detector, 'model') and state.detector.model is not None
            method_label = "YOLO(GPU)" if has_model else "TraditionalCV(fallback)"

        print(
            f"[VideoHandler] 帧#{self.frame_idx} | "
            f"FPS={fps:.1f} | "
            f"耗时={elapsed_ms:.0f}ms | "
            f"检测={stabilizer_stats.get('stable_results', 0)} | "
            f"跟踪={stabilizer_stats.get('tracked_objects', 0)} | "
            f"方法={method_label}"
        )

        self.last_log_time = time.time()


def _json_serializer(obj):
    """JSON 序列化器：处理 numpy 类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
