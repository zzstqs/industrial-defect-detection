"""
后端服务 - FastAPI + WebSocket 实时视频流

架构：模块化分离
├── state.py                 — 全局状态管理
├── detection_stabilizer.py  — 检测结果稳定器（消除闪烁）
├── services/
│   ├── detection_service.py — 缺陷检测服务
│   └── measurement_service.py — 尺寸测量服务
└── ws/
    └── video_handler.py     — WebSocket 视频流处理

提供：实时视频推流、缺陷检测API、尺寸测量API、前端页面服务、健康检查
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

from ..utils.config import BACKEND_HOST, BACKEND_PORT, FRONTEND_TITLE
from .state import get_state, update, snapshot, health_check
from .services import DetectionService, MeasurementService
from .ws.video_handler import VideoStreamHandler


# ============ FastAPI 应用 ============
app = FastAPI(
    title=FRONTEND_TITLE,
    version="2.0.0",
    description="工业视觉检测系统 — 模块化架构",
)

# CORS：允许前端独立部署时跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务实例（延迟初始化）
detection_service: Optional[DetectionService] = None
measurement_service: Optional[MeasurementService] = None


# ============ 启动事件 ============
@app.on_event("startup")
async def startup():
    """初始化所有服务模块"""
    global detection_service, measurement_service

    print("=" * 50)
    print("  工业视觉检测系统 — 启动中")
    print("=" * 50)

    # 检测服务
    detection_service = DetectionService()
    if not detection_service.init_yolo():
        print("[App] YOLO 不可用，降级到传统 CV 方法")
    detection_service.init_traditional()

    # 测量服务
    measurement_service = MeasurementService()
    measurement_service.init()

    # 不再把 service 覆盖到 state.detector/state.measurer
    # —— init_yolo()/init_traditional() 已正确注入 DefectDetector/TraditionalCVDetector
    # —— init() 已正确注入 DimensionMeasurer
    # VideoStreamHandler 通过构造函数接收 detection_service/measurement_service

    print("=" * 50)
    print("  所有服务初始化完成")
    print("=" * 50)


# ============ Pydantic 请求模型 ============
class MethodRequest(BaseModel):
    """检测方法切换请求"""
    method: str = Field(..., pattern="^(yolo|traditional)$")


class CVParamsUpdate(BaseModel):
    """传统CV参数批量更新"""
    canny_low: Optional[int] = Field(None, ge=10, le=200)
    canny_high: Optional[int] = Field(None, ge=30, le=300)
    morph_kernel: Optional[int] = Field(None, ge=1, le=9)
    area_min: Optional[int] = Field(None, ge=50, le=10000)
    area_max: Optional[int] = Field(None, ge=500, le=200000)
    thresh_block: Optional[int] = Field(None, ge=3, le=31)
    thresh_c: Optional[int] = Field(None, ge=0, le=20)
    lbp_anomaly_std: Optional[float] = Field(None, ge=1.0, le=5.0)
    fft_highpass_radius: Optional[int] = Field(None, ge=5, le=100)
    denoise_kernel: Optional[int] = Field(None, ge=1, le=15)
    sharpen_strength: Optional[float] = Field(None, ge=0, le=2.0)
    iou_threshold: Optional[float] = Field(None, ge=0.05, le=0.5)
    confidence_threshold: Optional[float] = Field(None, ge=0, le=0.9)
    max_detections: Optional[int] = Field(None, ge=10, le=200)
    enable_canny: Optional[bool] = None
    enable_threshold: Optional[bool] = None
    enable_lbp: Optional[bool] = None
    enable_fft: Optional[bool] = None
    enable_hsv: Optional[bool] = None
    use_clahe: Optional[bool] = None
    denoise_method: Optional[str] = None


# ============ WebSocket：实时视频流 ============
@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    """实时视频推流 — 整合检测+测量+稳定化"""
    handler = VideoStreamHandler(
        detection_service=detection_service,
        measurement_service=measurement_service,
    )
    await handler.run(websocket)


# ============ REST API：状态查询 ============
@app.get("/api/status")
def get_status():
    """获取系统运行状态"""
    return snapshot()


@app.get("/api/health")
def get_health():
    """系统健康检查（包含各模块可用性）"""
    hc = health_check()
    if not hc["camera_active"]:
        return JSONResponse(
            content={**hc, "status": "degraded"},
            status_code=200,
        )
    return hc


# ============ REST API：检测控制 ============
@app.post("/api/detection/toggle")
def toggle_detection():
    """切换缺陷检测 开/关"""
    state = get_state()
    new_state = not state.detection_enabled
    update(detection_enabled=new_state)

    # 切换时重置稳定器，避免残留结果
    if state.stabilizer:
        state.stabilizer.reset()

    return {"detection_enabled": new_state}


@app.post("/api/detection/method/{method}")
def set_detection_method(method: str):
    """切换检测方法：yolo / traditional"""
    if method not in ("yolo", "traditional"):
        raise HTTPException(status_code=400, detail="方法必须是 yolo 或 traditional")

    update(detection_method=method)

    # 切换方法时重置稳定器
    state = get_state()
    if state.stabilizer:
        state.stabilizer.reset()

    return {"detection_method": method}


# ============ REST API：测量控制 ============
@app.post("/api/measurement/toggle")
def toggle_measurement():
    """切换尺寸测量 开/关"""
    state = get_state()
    new_state = not state.measurement_enabled
    update(measurement_enabled=new_state)
    return {"measurement_enabled": new_state}


# ============ REST API：参数管理 ============
@app.get("/api/params")
def get_params():
    """获取传统CV和测量的当前参数"""
    state = get_state()
    result = {}

    if state.trad_cv_detector:
        p = state.trad_cv_detector.params
        result["cv"] = {
            "denoise_kernel": p.denoise_kernel,
            "denoise_method": p.denoise_method,
            "use_clahe": p.use_clahe,
            "sharpen_strength": p.sharpen_strength,
            "enable_canny": p.enable_canny,
            "canny_low": p.canny_low,
            "canny_high": p.canny_high,
            "morph_kernel": p.morph_kernel,
            "area_min": p.area_min,
            "area_max": p.area_max,
            "enable_threshold": p.enable_threshold,
            "thresh_block": p.thresh_block,
            "thresh_c": p.thresh_c,
            "enable_lbp": p.enable_lbp,
            "lbp_anomaly_std": p.lbp_anomaly_std,
            "enable_fft": p.enable_fft,
            "fft_highpass_radius": p.fft_highpass_radius,
            "enable_hsv": p.enable_hsv,
            "iou_threshold": p.iou_threshold,
            "confidence_threshold": p.confidence_threshold,
            "max_detections": p.max_detections,
        }

    if state.measurer:
        mp = state.measurer.params
        result["measure"] = {
            "use_subpixel": mp.use_subpixel,
            "canny_low": mp.canny_low,
            "canny_high": mp.canny_high,
            "show_measure_lines": mp.show_measure_lines,
            "circle_min_r": mp.circle_min_r,
            "circle_max_r": mp.circle_max_r,
        }

    return result


@app.post("/api/params/cv")
def update_cv_params(data: CVParamsUpdate):
    """实时更新传统CV参数（支持部分更新）"""
    state = get_state()
    if state.trad_cv_detector is None:
        raise HTTPException(status_code=503, detail="传统CV检测器未初始化")

    updated = []
    for key, value in data.model_dump(exclude_none=True).items():
        if hasattr(state.trad_cv_detector.params, key):
            setattr(state.trad_cv_detector.params, key, value)
            updated.append(key)

    return {"ok": True, "updated": updated}


@app.post("/api/params/measure")
def update_measure_params(data: dict):
    """实时更新测量参数"""
    state = get_state()
    if state.measurer is None:
        raise HTTPException(status_code=503, detail="测量模块未初始化")

    updated = []
    for key, value in data.items():
        if hasattr(state.measurer.params, key):
            setattr(state.measurer.params, key, value)
            updated.append(key)

    return {"ok": True, "updated": updated}


@app.get("/api/cv/channels")
def get_cv_channels():
    """获取传统CV各通道检测概况"""
    state = get_state()
    if state.trad_cv_detector is None:
        return {"channels": {}, "total_raw": 0}
    return state.trad_cv_detector.get_channel_summary()


@app.get("/api/stabilizer/stats")
def get_stabilizer_stats():
    """获取检测稳定器的运行统计"""
    state = get_state()
    if state.stabilizer:
        return state.stabilizer.get_stats()
    return {"error": "稳定器未初始化"}


# ============ 前端页面 ============
@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端页面"""
    html_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(
        content="<h1>前端文件未找到</h1><p>请确认 frontend/index.html 存在</p>",
        status_code=200,
    )


# ============ 全局异常处理 ============
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理，返回友好错误信息"""
    import traceback
    print(f"[App] 未处理的异常: {exc}")
    traceback.print_exc()
    return JSONResponse(
        content={"error": str(exc), "type": type(exc).__name__},
        status_code=500,
    )


# ============ 启动入口 ============
def start_server(host=BACKEND_HOST, port=BACKEND_PORT):
    """启动 FastAPI 服务"""
    print(f"\n{'=' * 50}")
    print(f"  工业视觉检测系统 v2.0 (模块化架构)")
    print(f"  Web 页面:   http://localhost:{port}")
    print(f"  WebSocket:  ws://localhost:{port}/ws/video")
    print(f"  API 文档:   http://localhost:{port}/docs")
    print(f"  健康检查:   http://localhost:{port}/api/health")
    print(f"{'=' * 50}\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        reload=False,  # 生产环境关闭热重载
    )


if __name__ == "__main__":
    start_server()
