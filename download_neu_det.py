#!/usr/bin/env python3
"""
NEU-DET 真实工业缺陷数据集 — 完整下载 + 转换 + 训练流水线
================================================================
6类缺陷: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
来源: Kaggle (kaustubhdikshit/neu-surface-defect-database)
共计: 1440 train + 360 val = 1800 images (200x200 grayscale)
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / \
    "kaustubhdikshit" / "neu-surface-defect-database" / "versions" / "1" / "NEU-DET"
YOLO_DIR = PROJECT_ROOT / "data" / "processed" / "neu_det_yolo"

CLASS_MAP = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5,
}
CLASS_NAMES = list(CLASS_MAP.keys())


def download_dataset():
    """从 Kaggle 下载 NEU-DET 数据集"""
    print("=" * 60)
    print("[Step 1] Downloading NEU-DET from Kaggle...")
    print("=" * 60)

    if CACHE_DIR.exists():
        imgs = list(CACHE_DIR.rglob("*.jpg"))
        if len(imgs) >= 100:
            print(f"  [SKIP] Already downloaded ({len(imgs)} images)")
            return True

    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "kaustubhdikshit/neu-surface-defect-database"
        )
        print(f"  Downloaded to: {path}")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def voc_to_yolo(size, box):
    """VOC XML bbox -> YOLO normalized bbox"""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh


def process_split(src_subdir: Path, split_name: str, stats: dict):
    """处理一个数据子集 (train/validation)"""
    img_base = src_subdir / "images"
    ann_base = src_subdir / "annotations"

    if not img_base.exists():
        print(f"  [SKIP] {split_name}: images not found")
        return

    split_img_dir = YOLO_DIR / "images" / split_name
    split_lbl_dir = YOLO_DIR / "labels" / split_name
    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 遍历所有类别子目录
    for class_name in CLASS_NAMES:
        class_img_dir = img_base / class_name
        if not class_img_dir.exists():
            continue

        for img_path in class_img_dir.glob("*.jpg"):
            img_name = img_path.stem  # e.g. "crazing_1"
            xml_path = ann_base / f"{img_name}.xml"

            if not xml_path.exists():
                stats["no_xml"] += 1
                continue

            try:
                # 解析 XML
                tree = ET.parse(xml_path)
                root = tree.getroot()

                size = root.find("size")
                w = int(size.find("width").text)
                h = int(size.find("height").text)

                yolo_lines = []
                for obj in root.iter("object"):
                    cls_name = obj.find("name").text.strip()
                    if cls_name not in CLASS_MAP:
                        continue

                    cls_id = CLASS_MAP[cls_name]
                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)

                    xc, yc, bw, bh = voc_to_yolo((w, h), (xmin, ymin, xmax, ymax))
                    yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                # 写 YOLO label
                lbl_path = split_lbl_dir / f"{img_name}.txt"
                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))

                # 复制图片
                dst_img = split_img_dir / f"{img_name}.jpg"
                shutil.copy2(img_path, dst_img)

                cls_id = CLASS_MAP.get(class_name, -1)
                stats[cls_id] = stats.get(cls_id, 0) + 1
                stats["total"] = stats.get("total", 0) + 1

            except Exception as e:
                stats["errors"] = stats.get("errors", 0) + 1
                if stats["errors"] <= 3:
                    print(f"  [WARN] {img_name}: {e}")


def convert_and_split():
    """转换标注 + 按 Kaggle 提供的 split 组织数据"""
    print("\n" + "=" * 60)
    print("[Step 2] Convert VOC XML -> YOLO TXT format...")
    print("=" * 60)

    # 清空输出目录
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    YOLO_DIR.mkdir(parents=True)

    stats = {"total": 0, "no_xml": 0, "errors": 0}

    for sub, name in [("train", "train"), ("validation", "val")]:
        src = CACHE_DIR / sub
        if src.exists():
            process_split(src, name, stats)
            print(f"  {name}: {stats['total']} images (and counting)")

    # 统计每个类别
    print(f"\n  Total: {stats['total']} images converted")
    print(f"  Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        cnt = stats.get(i, 0)
        print(f"    {name}: {cnt}")

    if stats["no_xml"]:
        print(f"  [WARN] {stats['no_xml']} images without XML")
    if stats["errors"]:
        print(f"  [WARN] {stats['errors']} conversion errors")

    return stats["total"] > 100


def create_yaml():
    """创建 data.yaml"""
    print("\n" + "=" * 60)
    print("[Step 3] Creating data.yaml...")
    print("=" * 60)

    yaml_content = f"""# NEU-DET Steel Surface Defect Detection Dataset
# 6 classes, 1800 grayscale images (200x200)

path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val

nc: 6
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""

    yaml_path = YOLO_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"  Created: {yaml_path}")

    # 验证
    train_imgs = list((YOLO_DIR / "images" / "train").glob("*.jpg"))
    val_imgs = list((YOLO_DIR / "images" / "val").glob("*.jpg"))
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val:   {len(val_imgs)} images")
    print(f"  Total: {len(train_imgs) + len(val_imgs)} images")
    return yaml_path


def main():
    print()
    print("=" * 60)
    print("  NEU-DET Industrial Defect Dataset Pipeline")
    print("  Source: Kaggle (kaustubhdikshit)")
    print("  Classes: crazing, inclusion, patches,")
    print("           pitted_surface, rolled-in_scale, scratches")
    print("=" * 60)

    if not download_dataset():
        return None

    if not convert_and_split():
        return None

    yaml_path = create_yaml()
    if not yaml_path:
        return None

    print()
    print("=" * 60)
    print("  [DONE] Dataset ready!")
    print(f"  YAML: {yaml_path}")
    print("=" * 60)
    return yaml_path


if __name__ == "__main__":
    yaml_path = main()
    if yaml_path:
        print(f"\nDATASET_YAML={yaml_path}")
    else:
        print("\n[DONE] Failed")
        sys.exit(1)
