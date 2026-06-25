"""
YOLOv11 工业缺陷检测模型训练脚本
支持：
- 自动检测数据集
- 断点续训
- 训练过程可视化
- 模型评估与导出
- 合成数据快速验证
"""

import sys
import argparse
from pathlib import Path
import yaml

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
import cv2


# ========== 默认配置 ==========
DEFAULT_CONFIG = {
    # 模型配置
    "model": "yolo11n.pt",   # 使用 YOLOv11 nano 版本（快，适合实时）
    # "model": "yolo11s.pt",  # 小版本（更准，稍慢）
    # "model": "yolo11m.pt",  # 中版本（更准确，需要更好 GPU）
    
    # 训练超参数
    "epochs": 100,              # 训练轮数（合成数据 50 足够，真实数据建议 100+）
    "batch": 16,                # 批次大小（根据显存调整，CPU 用 4~8）
    "imgsz": 640,              # 输入图片尺寸
    "device": "cpu",            # 设备：cpu / 0（GPU 0）/ 0,1（多 GPU）
    
    # 数据增强
    "augment": True,            # 启用数据增强
    "hsv_h": 0.015,           # 色调增强
    "hsv_s": 0.7,             # 饱和度增强
    "hsv_v": 0.4,             # 亮度增强
    "flipud": 0.2,             # 上下翻转概率
    "fliplr": 0.5,            # 左右翻转概率
    "mosaic": 1.0,             # Mosaic 增强（4 图拼接）
    "mixup": 0.15,             # MixUp 增强
    
    # 优化器
    "optimizer": "AdamW",       # 优化器（SGD / Adam / AdamW）
    "lr0": 0.001,              # 初始学习率
    "weight_decay": 0.0005,    # 权重衰减
    "warmup_epochs": 3.0,      # 预热轮数
    
    # 输出
    "project": str(PROJECT_ROOT / "models" / "runs"),
    "name": "defect_detection",
    "exist_ok": True,
    "pretrained": True,
    "save": True,
    "save_period": 10,          # 每 N 轮保存一次
    "val": True,                # 每轮都验证
    "plots": True,              # 生成训练曲线图
}


def find_dataset_yaml() -> Path:
    """自动查找 data.yaml 文件"""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    # 优先使用真实数据集，其次合成数据
    candidates = [
        processed_dir / "neu_det_yolo" / "data.yaml",
        processed_dir / "synthetic_yolo" / "data.yaml",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            print(f"[数据集] 找到配置文件: {candidate}")
            return candidate
    
    print("[数据集] 未找到 data.yaml 文件")
    print("=" * 60)
    print("请先运行数据集准备脚本：")
    print(f"  cd {PROJECT_ROOT}")
    print(f"  python data/prepare_dataset.py")
    print("=" * 60)
    return None


def check_dataset(yaml_path: Path) -> bool:
    """检查数据集是否完整"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config["path"])
    train_dir = data_dir / config["train"]
    val_dir = data_dir / config["val"]
    
    if not train_dir.exists():
        print(f"[错误] 训练集目录不存在: {train_dir}")
        return False
    
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    print(f"[检查] 训练集图片数量: {len(train_images)}")
    
    if len(train_images) == 0:
        print("[错误] 训练集没有图片！")
        return False
    
    if len(train_images) < 10:
        print("[警告] 训练集图片数量较少，建议至少 50 张以上")
    
    return True


def train_model(data_yaml: Path, args: dict):
    """训练 YOLOv11 模型"""
    print("=" * 60)
    print("开始训练 YOLOv11 缺陷检测模型")
    print("=" * 60)
    
    # 加载模型（预训练权重）
    model_name = args.get("model", DEFAULT_CONFIG["model"])
    print(f"[模型] 加载: {model_name}")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"[错误] 无法加载模型 {model_name}: {e}")
        print("[提示] 如果是首次运行，会自动下载预训练权重")
        print("[提示] 如果下载失败，可以手动下载后放到 ~/.config/Ultralytics/ 目录")
        return None
    
    # 构建训练参数
    train_args = {**DEFAULT_CONFIG, **args}
    train_args["data"] = str(data_yaml)
    
    # 移除不在 YOLO 训练接口中的参数
    yolo_args = {k: v for k, v in train_args.items() 
                 if k not in ["augment", "device"] or k == "device"}
    
    print(f"[训练] 参数:")
    print(f"  数据配置: {data_yaml}")
    print(f"  训练轮数: {train_args['epochs']}")
    print(f"  批次大小: {train_args['batch']}")
    print(f"  图片尺寸: {train_args['imgsz']}")
    print(f"  设备:     {train_args['device']}")
    print(f"  优化器:   {train_args['optimizer']}")
    print()
    
    # 开始训练
    try:
        results = model.train(**train_args)
        print()
        print("=" * 60)
        print("训练完成！")
        print(f"最佳模型: {results.save_dir / 'weights' / 'best.pt'}")
        print(f"最后模型: {results.save_dir / 'weights' / 'last.pt'}")
        print("=" * 60)
        return results
    except Exception as e:
        print(f"[错误] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model(model_path: Path, data_yaml: Path):
    """评估模型性能"""
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        return
    
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml))
    
    print(f"[评估] mAP50:   {metrics.box.map50:.4f}")
    print(f"[评估] mAP50-95: {metrics.box.map:.4f}")
    print(f"[评估] 精确率:  {metrics.box.precision:.4f}")
    print(f"[评估] 召回率:  {metrics.box.recall:.4f}")
    
    return metrics


def export_model(model_path: Path, format: str = "onnx"):
    """导出模型为其他格式"""
    print("=" * 60)
    print(f"导出模型为 {format.upper()} 格式")
    print("=" * 60)
    
    model = YOLO(str(model_path))
    success = model.export(format=format)
    
    if success:
        print(f"[导出] 成功！文件: {model_path.with_suffix(f'.{format}')}")
    else:
        print(f"[导出] 失败")
    
    return success


def quick_train_synthetic():
    """快速训练（使用合成数据，验证整个流程）"""
    print("=" * 60)
    print("快速训练模式 - 使用合成数据")
    print("=" * 60)
    
    # 准备合成数据
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "data"))
    
    # 检查是否已有合成数据
    synthetic_yaml = PROJECT_ROOT / "data" / "processed" / "synthetic_yolo" / "data.yaml"
    
    if not synthetic_yaml.exists():
        print("[信息] 合成数据不存在，正在生成...")
        from prepare_dataset import prepare_synthetic_data
        yolo_dir = prepare_synthetic_data()
        synthetic_yaml = yolo_dir / "data.yaml"
    
    # 使用 CPU 训练（更快启动）
    args = {
        "model": "yolo11n.pt",
        "epochs": 30,           # 合成数据不需要太多轮
        "batch": 8,              # CPU 用小 batch
        "imgsz": 640,
        "device": "cpu",         # 如果有 GPU 改为 "0"
        "optimizer": "AdamW",
        "lr0": 0.001,
        "name": "synthetic_quick",
    }
    
    results = train_model(synthetic_yaml, args)
    
    if results:
        best_model = Path(results.save_dir) / "weights" / "best.pt"
        if best_model.exists():
            print()
            print("[完成] 快速训练完成！模型已保存到:")
            print(f"  {best_model}")
            print()
            print("[下一步] 将模型集成到实时检测系统:")
            print(f"  1. 复制模型到 models/ 目录:")
            print(f"     copy {best_model} {PROJECT_ROOT / 'models' / 'best.pt'}")
            print(f"  2. 修改 src/detection/defect_detector.py 中的 MODEL_PATH")
            print(f"  3. 重启后端服务，选择『YOLO』方法")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 工业缺陷检测模型训练")
    parser.add_argument("--data", type=str, default=None, help="data.yaml 路径（不指定则自动查找）")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="模型名称或路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图片尺寸")
    parser.add_argument("--device", type=str, default="cpu", help="设备（cpu / 0 / 0,1）")
    parser.add_argument("--quick", action="store_true", help="快速训练模式（合成数据）")
    parser.add_argument("--eval", type=str, default=None, help="评估指定模型")
    parser.add_argument("--export", type=str, default=None, help="导出指定模型（提供模型路径）")
    parser.add_argument("--format", type=str, default="onnx", help="导出格式（onnx / torchscript）")
    
    args = parser.parse_args()
    
    # 快速训练模式
    if args.quick:
        quick_train_synthetic()
        return
    
    # 评估模式
    if args.eval:
        model_path = Path(args.eval)
        data_yaml = find_dataset_yaml()
        if data_yaml:
            evaluate_model(model_path, data_yaml)
        return
    
    # 导出模式
    if args.export:
        export_model(Path(args.export), args.format)
        return
    
    # 正常训练模式
    data_yaml = Path(args.data) if args.data else find_dataset_yaml()
    if not data_yaml or not data_yaml.exists():
        print("[错误] 未找到数据集配置文件")
        print("请运行: python data/prepare_dataset.py")
        return
    
    if not check_dataset(data_yaml):
        return
    
    # 构建训练参数
    train_args = {
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
    }
    
    train_model(data_yaml, train_args)


if __name__ == "__main__":
    main()
