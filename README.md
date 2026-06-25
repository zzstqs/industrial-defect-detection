# 工业视觉检测系统
 
> 基于 **YOLOv11 + OpenCV** 的实时工业缺陷检测与尺寸测量系统

![Python](https://img.shields.io/badge/Python-3.13-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![YOLO](https://img.shields.io/badge/YOLO-v11-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)

## 功能特性

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| 🔍 缺陷检测 | 实时识别表面裂纹、夹杂、划痕等缺陷 | YOLOv11 + 传统CV（Canny/轮廓）双引擎 |
| 📏 尺寸测量 | 实时测量物体宽高面积（mm） | OpenCV 相机标定 + 参考物比例法 |
| 📡 实时视频流 | 浏览器查看检测结果，毫秒级延迟 | FastAPI + WebSocket |
| 🖥️ 前端面板 | 检测结果可视化、参数实时调节 | 原生 HTML/JS/CSS，响应式设计 |
| 🧠 已训练模型 | NEU-DET 数据集训练，mAP@50=72.7% | 6类缺陷：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches |

## 项目结构（v2.0 模块化架构）

```
shibie/
├── run.py                          # 启动入口
├── train.py                        # YOLOv11 训练脚本
├── download_neu_det.py             # NEU-DET 数据集下载工具
├── requirements.txt
├── src/
│   ├── detection/
│   │   └── defect_detector.py      # YOLO + 传统CV 缺陷检测
│   ├── measurement/
│   │   └── dimension_measurer.py   # 尺寸测量（标定 + 参考物）
│   ├── utils/
│   │   ├── config.py               # 全局配置
│   │   └── camera.py               # 摄像头采集
│   └── backend/                    # FastAPI 后端（模块化）
│       ├── app.py                  # 路由 + 启动
│       ├── state.py                # 全局状态管理（线程安全）
│       ├── detection_stabilizer.py # 检测结果时间滤波
│       ├── services/
│       │   ├── detection_service.py
│       │   └── measurement_service.py
│       └── ws/
│           └── video_handler.py    # WebSocket 帧处理管线
├── frontend/
│   └── index.html                  # Web 前端（设计系统 + 响应式）
├── models/
│   └── best.pt                     # 训练好的 NEU-DET 模型
└── tests/
```

## 快速开始

### 1. 环境要求

- Python 3.11+
- CUDA（可选，GPU 推理加速）
- 摄像头（USB 或笔记本内置）

### 2. 创建虚拟环境 & 安装依赖

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 下载 NEU-DET 数据集（可选）

数据集已 `.gitignore`，如需重新训练模型：

```bash
python download_neu_det.py
```

### 4. 运行系统

```bash
# 启动 Web 服务
python run.py
# 浏览器访问 http://localhost:8000
# API 文档 http://localhost:8000/docs
```

### 5. 训练模型（可选）

```bash
# 使用已下载的 NEU-DET 数据集训练
python train.py
```

## 模型信息

| 项目 | 详情 |
|------|------|
| 基础模型 | YOLOv11n |
| 训练数据 | NEU-DET（东北大学钢材表面缺陷数据集） |
| 训练轮数 | 50 epochs |
| mAP@50 | 72.7% |
| 推理设备 | GPU（NVIDIA RTX 5060, PyTorch 2.11+cu128） |
| 类别 | crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches |

## 系统架构

```
摄像头 / 视频文件
       ↓
  OpenCV 帧采集
       ↓
┌──────────────────────┐
│   算法处理层 (线程池)  │
│  ┌────────────────┐  │
│  │ 缺陷检测        │── YOLOv11 / 传统CV 双引擎
│  │ DetectionStabilizer │── 时间滤波防闪烁
│  ├────────────────┤  │
│  │ 尺寸测量        │── OpenCV 标定 + 参考物
│  └────────────────┘  │
└──────────┬───────────┘
           ↓
   FastAPI 后端 (asyncio)
   WebSocket 实时推送
           ↓
   浏览器前端 (增量渲染)
   实时视频 + 数据面板
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | Web 前端页面 |
| `GET` | `/api/health` | 系统健康检查 |
| `POST` | `/api/detection/yolo/init` | 初始化 YOLO 模型 |
| `POST` | `/api/detection/yolo/toggle` | 切换 YOLO 检测开关 |
| `POST` | `/api/detection/traditional/toggle` | 切换传统CV检测 |
| `POST` | `/api/detection/cv/params` | 调整 CV 检测参数 |
| `POST` | `/api/measurement/toggle` | 切换测量开关 |
| `POST` | `/api/measurement/calibrate` | 触发相机标定 |
| `POST` | `/api/measurement/reference` | 设置参考物参数 |
| `WS` | `/ws/video` | WebSocket 视频流 |

## License

MIT
