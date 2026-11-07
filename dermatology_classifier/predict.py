"""
疾病分类推理：支持单张图片或整个文件夹批量预测。

用法:
    python predict.py --image path/to/img.jpg --checkpoint best.pth
    python predict.py --dir   path/to/folder  --checkpoint best.pth
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
    DISEASE_CLASSES, DISEASE_CLASSES_ZH, NUM_DISEASE_CLASSES, CHECKPOINT_DIR,
)
from transforms import get_val_transforms
from model import build_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="疾病三分类推理（单图或文件夹批量）")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="单张图片路径")
    g.add_argument("--dir", type=str, help="文件夹路径，预测其中所有图片")
    p.add_argument("--checkpoint", type=str, default="best.pth")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_image(path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_images(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def predict_one(model, img_path, tf, device):
    image = load_image(str(img_path))
    tensor = tf(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs


def print_single(pred, probs):
    print("\n" + "=" * 50)
    print("疾病分类预测")
    print("=" * 50)
    for i, name in enumerate(DISEASE_CLASSES):
        marker = "  <-- 预测" if i == pred else ""
        print(f"  {name:22s} ({DISEASE_CLASSES_ZH[name]}): {probs[i]:.4f}{marker}")
    print(f"\n预测结果: {DISEASE_CLASSES[pred]} ({DISEASE_CLASSES_ZH[DISEASE_CLASSES[pred]]})"
          f"  置信度 {probs[pred]:.4f}")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = CHECKPOINT_DIR / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    image_size = cfg.get("image_size", 224)
    backbone = cfg.get("backbone", "convnext_tiny")

    model = build_model(
        backbone_name=backbone, num_classes=NUM_DISEASE_CLASSES, pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tf = get_val_transforms(image_size)

    # ---- 单张图片 ----
    if args.image:
        pred, probs = predict_one(model, args.image, tf, device)
        print_single(pred, probs)
        return

    # ---- 文件夹批量 ----
    folder = Path(args.dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    image_paths = list_images(folder)
    if not image_paths:
        print(f"文件夹 {folder} 内没有找到图片（支持: {', '.join(sorted(IMAGE_EXTS))}）")
        return

    print("\n" + "=" * 50)
    print(f"批量预测: {folder}  ({len(image_paths)} 张)")
    print("=" * 50)

    counts = Counter()
    for p in image_paths:
        pred, probs = predict_one(model, p, tf, device)
        name_en = DISEASE_CLASSES[pred]
        counts[name_en] += 1
        print(f"  {p.name:30s} -> {name_en:22s} ({DISEASE_CLASSES_ZH[name_en]})"
              f"  置信度 {probs[pred]:.4f}")

    print("\n" + "-" * 50)
    print("汇总:")
    for name in DISEASE_CLASSES:
        print(f"  {name:22s}: {counts.get(name, 0)} 张")
    print(f"  合计: {len(image_paths)} 张")


if __name__ == "__main__":
    main()
