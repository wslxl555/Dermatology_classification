"""
可解释性推理：整图疾病分类 + 病灶检测，联合输出。

对一张图同时跑：
  1) 疾病分类器（outputs/checkpoints/best.pth）→ 病种 + 概率
  2) 病灶检测器（outputs/checkpoints/lesion_detector.pth）→ 检出哪些病灶、几个、在哪

并给出「关键病灶是否与预测一致」的解释说明。

用法:
    python detector/predict_interpretable.py --image path/to/img.jpg
    python detector/predict_interpretable.py --dir path/to/folder
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch

from config import CHECKPOINT_DIR, DISEASE_CLASSES, DISEASE_CLASSES_ZH
from transforms import get_val_transforms
from model import build_model
from detector.detection_model import (
    DETECTION_CLASSES, DETECTION_CLASSES_ZH, build_detector,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 每种疾病的关键病灶（用于解释一致性）
DISEASE_KEY_LESIONS = {
    "bullous_pemphigoid": ["bulla", "vesicle"],
    "eczema": ["scale", "vesicle", "erosion", "crust"],
    "urticaria": ["papule", "plaque", "patch"],
}


def parse_args():
    p = argparse.ArgumentParser(description="可解释性推理（分类 + 检测）")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="单张图片路径")
    g.add_argument("--dir", type=str, help="文件夹路径")
    p.add_argument("--classifier", type=str, default=str(CHECKPOINT_DIR / "best.pth"))
    p.add_argument("--detector", type=str,
                   default=str(CHECKPOINT_DIR / "lesion_detector.pth"))
    p.add_argument("--score_thresh", type=float, default=0.3)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_classifier(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    image_size = cfg.get("image_size", 224)
    backbone = cfg.get("backbone", "convnext_tiny")
    model = build_model(backbone_name=backbone, num_classes=len(DISEASE_CLASSES),
                        pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, get_val_transforms(image_size), image_size


def load_detector(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = ckpt.get("detection_head", "fcos")
    image_size = ckpt.get("image_size", 512)
    model = build_detector(head).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, image_size


def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def classify_image(cls_model, cls_tf, device, image):
    with torch.no_grad():
        tensor = cls_tf(image=image)["image"].unsqueeze(0).to(device)
        probs = torch.softmax(cls_model(tensor), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs


def detect_image(det_model, det_size, device, image):
    h, w = image.shape[:2]
    resized = cv2.resize(image, (det_size, det_size))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        pred = det_model([tensor])[0]
    # 把框坐标映射回原图
    sx, sy = w / det_size, h / det_size
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    boxes = boxes * np.array([sx, sy, sx, sy])
    return boxes, labels, scores


def interpret(pred_disease, det_classes_present):
    key = DISEASE_KEY_LESIONS.get(DISEASE_CLASSES[pred_disease], [])
    hits = [k for k in key if k in det_classes_present]
    if hits:
        return f"检出关键病灶 {('、'.join(hits))}，与预测一致"
    if det_classes_present:
        return f"未检出关键病灶，检出 {('、'.join(det_classes_present))}，建议人工复核"
    return "未检出病灶"


def report_image(image_path, cls_model, cls_tf, det_model, det_size, device,
                 score_thresh):
    image = load_image(str(image_path))
    pred, probs = classify_image(cls_model, cls_tf, device, image)
    boxes, labels, scores = detect_image(det_model, det_size, device, image)

    print("\n" + "=" * 60)
    print(f"图片: {Path(image_path).name}")
    print("=" * 60)
    print("疾病分类:")
    for i, name in enumerate(DISEASE_CLASSES):
        marker = "  <-- 预测" if i == pred else ""
        print(f"  {name:22s} ({DISEASE_CLASSES_ZH[name]}): {probs[i]:.4f}{marker}")

    print(f"\n病灶检测（置信度>{score_thresh}）:")
    from collections import Counter
    keep = scores >= score_thresh
    det_classes = set()
    if keep.sum() == 0:
        print("  （未检出病灶）")
    else:
        counts = Counter()
        for lab, sc in zip(labels[keep], scores[keep]):
            name = DETECTION_CLASSES[lab - 1]  # 背景=0，前景 1..N
            counts[name] += 1
            det_classes.add(name)
        for name in DETECTION_CLASSES:
            if counts.get(name):
                print(f"  {name:12s} ({DETECTION_CLASSES_ZH[name]}): "
                      f"{counts[name]} 处")

    print(f"\n解释: {interpret(pred, det_classes)}")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cls_model, cls_tf, _ = load_classifier(args.classifier, device)
    det_model, det_size = load_detector(args.detector, device)
    print(f"分类器: {args.classifier}")
    print(f"检测器: {args.detector} (输入 {det_size}px)")

    if args.image:
        report_image(args.image, cls_model, cls_tf, det_model, det_size,
                     device, args.score_thresh)
        return

    folder = Path(args.dir)
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    print(f"\n批量推理 {len(paths)} 张")
    for p in paths:
        report_image(p, cls_model, cls_tf, det_model, det_size,
                     device, args.score_thresh)


if __name__ == "__main__":
    main()
