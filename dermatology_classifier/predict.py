"""
单张图片疾病分类推理。

用法:
    python predict.py --image path/to/img.jpg --checkpoint best.pth
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
    DISEASE_CLASSES, DISEASE_CLASSES_ZH, NUM_DISEASE_CLASSES, CHECKPOINT_DIR,
)
from transforms import get_val_transforms
from model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="单图疾病三分类推理")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="best.pth")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_image(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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

    image = load_image(args.image)
    tf = get_val_transforms(image_size)
    tensor = tf(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred = int(probs.argmax())
    print("\n" + "=" * 50)
    print("疾病分类预测")
    print("=" * 50)
    for i, name in enumerate(DISEASE_CLASSES):
        marker = "  <-- 预测" if i == pred else ""
        print(f"  {name:22s} ({DISEASE_CLASSES_ZH[name]}): {probs[i]:.4f}{marker}")
    print(f"\n预测结果: {DISEASE_CLASSES[pred]} ({DISEASE_CLASSES_ZH[DISEASE_CLASSES[pred]]})"
          f"  置信度 {probs[pred]:.4f}")


if __name__ == "__main__":
    main()
