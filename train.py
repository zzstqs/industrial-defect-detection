"""
工业缺陷检测 - 一键训练脚本
用法：
  python train.py              # 快速训练（合成数据，约5~10分钟）
  python train.py --real      # 使用真实数据集训练（需先准备数据）
  python train.py --eval      # 评估已有模型
  python train.py --export    # 导出模型为 ONNX 格式
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 颜色输出（兼容 Windows）
GREEN = "[OK] "
YELLOW = "[!] "
BLUE = "[INFO] "
RED = "[ERR] "
RESET = ""


def print_header(text):
    print(f"\n{GREEN}{'=' * 60}{RESET}")
    print(f"{GREEN}{text}{RESET}")
    print(f"{GREEN}{'=' * 60}{RESET}\n")


def step1_prepare_data():
    """Step 1: 准备数据集"""
    print_header("Step 1: 准备数据集")
    
    from data.prepare_dataset import prepare_synthetic_data, find_dataset_yaml
    
    # 检查是否已有数据集
    yaml_path = find_dataset_yaml()
    if yaml_path:
        print(f"{GREEN}[✓] 已找到数据集: {yaml_path}{RESET}")
        return yaml_path
    
    # 生成合成数据
    print(f"{YELLOW}[!] 未找到数据集，生成合成数据...{RESET}")
    yolo_dir = prepare_synthetic_data()
    yaml_path = yolo_dir / "data.yaml"
    
    print(f"{GREEN}[✓] 合成数据集已生成: {yaml_path}{RESET}")
    return yaml_path


def step2_train(data_yaml: Path, use_gpu: bool = False, epochs: int = 30):
    """Step 2: 训练模型"""
    print_header("Step 2: 训练 YOLOv11 模型")
    
    # 设置 Ultralytics 缓存目录到项目内（避免沙箱拦截 AppData）
    ultralytics_cache_dir = PROJECT_ROOT / ".ultralytics_cache"
    ultralytics_cache_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_cache_dir)
    # 同时设置 NVIDIA 缓存目录
    os.environ["CUDA_CACHE_PATH"] = str(ultralytics_cache_dir / "cuda_cache")
    print(f"{BLUE}[信息] Ultralytics 缓存目录: {ultralytics_cache_dir}{RESET}")
    
    from ultralytics import YOLO
    
    device = "0" if use_gpu else "cpu"
    print(f"{BLUE}[信息] 使用设备: {device}{RESET}")
    print(f"{BLUE}[信息] 训练轮数: {epochs}{RESET}")
    print(f"{BLUE}[信息] 数据集: {data_yaml}{RESET}")
    print()
    
    # 加载模型
    model = YOLO("yolo11n.pt")
    print(f"{GREEN}[✓] 模型加载成功{RESET}")
    
    # 开始训练
    print(f"{YELLOW}[开始] 训练进行中，请耐心等待...{RESET}")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=4 if device == "cpu" else 8,   # 减小 batch，避免 OOM
        imgsz=640,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        project=str(PROJECT_ROOT / "models" / "runs"),
        name="defect_detection",
        exist_ok=True,
        pretrained=True,
        save=True,
        val=True,
        plots=True,
        verbose=True,
        workers=0,            # 关键：设为 0 避免多进程崩溃
    )
    
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n{GREEN}[✓] 训练完成！{RESET}")
    print(f"{GREEN}[✓] 最佳模型: {best_model}{RESET}")
    
    return best_model


def step3_evaluate(model_path: Path, data_yaml: Path):
    """Step 3: 评估模型"""
    print_header("Step 3: 评估模型性能")
    
    from ultralytics import YOLO
    
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml))
    
    print(f"{BLUE}评估结果:{RESET}")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  精确率:    {metrics.box.mp:.4f}")
    print(f"  召回率:    {metrics.box.mr:.4f}")
    
    return metrics


def step4_deploy(model_path: Path):
    """Step 4: 部署模型到系统"""
    print_header("Step 4: 部署模型到检测系统")
    
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    deployed_path = models_dir / "best.pt"
    
    # 复制模型
    import shutil
    shutil.copy(str(model_path), str(deployed_path))
    
    print(f"{GREEN}[✓] 模型已部署到: {deployed_path}{RESET}")
    print(f"{BLUE}[信息] 现在可以启动系统使用训练好的模型了！{RESET}")
    print()
    print(f"{YELLOW}启动命令:{RESET}")
    print(f"  python run.py")
    print(f"  然后打开浏览器: http://localhost:8000")
    print(f"  点击「缺陷检测」按钮即可使用训练好的模型！")
    print()
    
    # 更新 config.py 中的 TRAINED_MODEL_PATH
    config_path = PROJECT_ROOT / "src" / "utils" / "config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 替换 TRAINED_MODEL_PATH
    import re
    new_line = 'TRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")  # 训练好的模型路径'
    content = re.sub(
        r'TRAINED_MODEL_PATH = .*',
        new_line,
        content
    )
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"{GREEN}[✓] 已更新配置文件，下次启动自动加载训练好的模型{RESET}")


def main():
    parser = argparse.ArgumentParser(description="工业缺陷检测 - 一键训练")
    parser.add_argument("--real", action="store_true", help="使用真实数据集（需先手动下载）")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数（默认30）")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 训练（默认自动检测 GPU）")
    parser.add_argument("--eval", type=str, default=None, help="评估指定模型路径")
    parser.add_argument("--export", type=str, default=None, help="导出模型为 ONNX 格式")
    args = parser.parse_args()
    
    print(f"{BLUE}")
    print("  █████╗ ██╗   ██╗███████╗██████╗ ")
    print("  ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗")
    print("  ██████╔╝ ╚████╔╝ █████╗  ██████╔╝")
    print("  ██╔═══╝   ╚██╔╝  ██╔══╝  ██╔══██╗")
    print("  ██║        ██║   ███████╗██║  ██║")
    print("  ╚═╝        ╚═╝   ╚══════╝╚═╝  ╚═╝")
    print("       工业缺陷检测 - YOLOv11 训练工具")
    print(f"{RESET}")
    
    # 自动检测 GPU
    import torch
    use_gpu = torch.cuda.is_available() and not args.cpu
    device_name = f"GPU ({torch.cuda.get_device_name(0)})" if use_gpu else "CPU"
    print(f"{BLUE}[信息] 使用设备: {device_name}{RESET}")
    
    # 评估模式
    if args.eval:
        model_path = Path(args.eval)
        if not model_path.exists():
            print(f"{RED}[错误] 模型文件不存在: {model_path}{RESET}")
            return
        data_yaml = step1_prepare_data()
        step3_evaluate(model_path, data_yaml)
        return
    
    # 导出模式
    if args.export:
        from ultralytics import YOLO
        model_path = Path(args.export)
        if not model_path.exists():
            print(f"{RED}[错误] 模型文件不存在: {model_path}{RESET}")
            return
        model = YOLO(str(model_path))
        export_path = model.export(format="onnx")
        print(f"{GREEN}[✓] 模型已导出: {export_path}{RESET}")
        return
    
    # 完整训练流程
    print(f"{YELLOW}[提示] 这将进行完整的训练流程：{RESET}")
    print(f"  1. 准备数据集（合成数据，无需手动标注）")
    print(f"  2. 训练 YOLOv11 模型（约 5~10 分钟）")
    print(f"  3. 评估模型性能")
    print(f"  4. 部署模型到系统")
    print()
    
    try:
        # Step 1: 准备数据
        data_yaml = step1_prepare_data()
        
        # Step 2: 训练
        best_model = step2_train(data_yaml, use_gpu=use_gpu, epochs=args.epochs)
        
        # Step 3: 评估
        if best_model.exists():
            step3_evaluate(best_model, data_yaml)
            
            # Step 4: 部署
            step4_deploy(best_model)
        else:
            print(f"{RED}[错误] 最佳模型未找到，训练可能失败{RESET}")
        
        print(f"\n{GREEN}{'=' * 60}{RESET}")
        print(f"{GREEN}  训练流程全部完成！{RESET}")
        print(f"{GREEN}{'=' * 60}{RESET}\n")
        
        print(f"{YELLOW}下一步：{RESET}")
        print(f"  1. 启动系统: {BLUE}python run.py{RESET}")
        print(f"  2. 打开浏览器: {BLUE}http://localhost:8000{RESET}")
        print(f"  3. 点击「缺陷检测」查看训练好的模型效果！")
        
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[取消] 训练已取消{RESET}")
    except Exception as e:
        print(f"\n{RED}[错误] 训练失败: {e}{RESET}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
