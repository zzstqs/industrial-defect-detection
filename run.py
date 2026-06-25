"""
项目启动脚本
使用方法：
    python run.py              # 启动完整 Web 服务（推荐）
    python run.py --mode demo  # 启动离线演示模式（摄像头实时预览 + 控制台输出）
    python run.py --mode test  # 运行单元测试
"""

import argparse
import sys
import os

# 将项目根目录加入 sys.path，确保包导入正常
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def start_web_server():
    """启动 FastAPI Web 服务"""
    print("=" * 50)
    print("  工业视觉检测系统 - Web 服务模式")
    print("=" * 50)
    from src.backend.app import start_server
    start_server()


def start_demo_mode():
    """离线演示模式：打开摄像头，实时显示检测结果（无需浏览器）"""
    print("=" * 50)
    print("  工业视觉检测系统 - 离线演示模式")
    print("  按 q 退出")
    print("=" * 50)

    import cv2
    from src.utils.camera import CameraCapture
    from src.detection.defect_detector import DefectDetector, detect_defects_traditional
    from src.measurement.dimension_measurer import DimensionMeasurer

    # 初始化
    try:
        detector = DefectDetector()
        print("[Demo] YOLO 模型已加载")
    except Exception as e:
        detector = None
        print(f"[Demo] YOLO 加载失败，使用传统 CV: {e}")

    measurer = DimensionMeasurer()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Demo] 无法打开摄像头！请检查摄像头连接。")
        return

    enable_detection = True
    enable_measurement = True
    method = "traditional"  # 离线模式默认用传统方法（无需训练）

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = frame.copy()

        # 缺陷检测
        if enable_detection:
            if detector and method == "yolo":
                result, detections = detector.process_frame(frame)
                for d in detections:
                    print(f"  [检测] {d['class']} 置信度:{d['confidence']:.2f}")
            else:
                result, _ = detect_defects_traditional(frame)

        # 尺寸测量
        if enable_measurement:
            ref_width, result = measurer.find_reference_object(result)
            if ref_width:
                print(f"  [测量] 参考物已识别，像素比例: {measurer.pixel_per_mm:.2f} px/mm")

        # 显示
        cv2.imshow("Industrial Vision Detection - Press q to quit", result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            enable_detection = not enable_detection
            print(f"  [控制] 缺陷检测: {'开' if enable_detection else '关'}")
        if key == ord('m'):
            enable_measurement = not enable_measurement
            print(f"  [控制] 尺寸测量: {'开' if enable_measurement else '关'}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Demo] 演示结束")


def run_tests():
    """运行基础测试"""
    print("=" * 50)
    print("  工业视觉检测系统 - 单元测试")
    print("=" * 50)

    # 测试1：导入检查
    print("\n[Test 1] 模块导入检查...")
    try:
        from src.utils.config import PROJECT_ROOT
        from src.utils.camera import CameraCapture
        from src.detection.defect_detector import DefectDetector
        from src.measurement.dimension_measurer import DimensionMeasurer
        print("  PASSED: 所有模块导入成功")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # 测试2：生成模拟图像并测试检测
    print("\n[Test 2] 模拟图像缺陷检测...")
    import numpy as np
    dummy = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    try:
        from src.detection.defect_detector import detect_defects_traditional
        result, regions = detect_defects_traditional(dummy)
        print(f"  PASSED: 传统 CV 检测完成，检测到 {len(regions)} 个区域")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # 测试3：尺寸测量
    print("\n[Test 3] 尺寸测量模块...")
    try:
        from src.measurement.dimension_measurer import DimensionMeasurer, quick_measure
        m = DimensionMeasurer()
        m.set_pixel_per_mm(100, 50)  # 100px = 50mm
        r = m.measure_object([0, 0, 200, 100], "test")
        print(f"  PASSED: 测量完成，宽度 = {r['width_mm']} mm")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    print("\n所有测试通过！")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="工业视觉检测系统启动脚本")
    parser.add_argument("--mode", choices=["web", "demo", "test"], default="web",
                        help="运行模式: web(默认) / demo / test")
    args = parser.parse_args()

    if args.mode == "web":
        start_web_server()
    elif args.mode == "demo":
        start_demo_mode()
    elif args.mode == "test":
        success = run_tests()
        sys.exit(0 if success else 1)
