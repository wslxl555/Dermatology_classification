"""
训练分割式检测头（attention_unet / mamba / sam_adapter / transunet / transcascade）。

这些头是「分割式检测」（bbox 渲染成 mask 做弱监督分割，推理时连通域转回框），
已在 dermatology_detection 工程实现。由于两个工程有顶层 config/models 等模块名冲突，
这里用【子进程】调用 dermatology_detection/train_labeled.py，避免 import 冲突。

用法:
    python detector/train_segmentation.py --head attention_unet --epochs 20 --offline
"""

import argparse
import subprocess
import sys
from pathlib import Path

DET_ROOT = Path(__file__).resolve().parent.parent.parent / "dermatology_detection"
SEGMENTATION_HEAD_NAMES = [
    "attention_unet", "mamba", "sam_adapter", "transunet", "transcascade",
]


def parse_args():
    p = argparse.ArgumentParser(description="训练分割式检测头（子进程桥接）")
    p.add_argument("--head", type=str, default="attention_unet",
                   choices=SEGMENTATION_HEAD_NAMES)
    p.add_argument("--backbone", type=str, default="convnext_tiny")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    script = DET_ROOT / "train_labeled.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 {script}（dermatology_detection 工程缺失）")

    cmd = [sys.executable, str(script),
           "--detection_head", args.head,
           "--backbone", args.backbone,
           "--image_size", str(args.image_size),
           "--batch_size", str(args.batch_size),
           "--epochs", str(args.epochs),
           "--lr", str(args.lr)]
    if args.offline:
        cmd.append("--offline")

    print(f"子进程调用: {' '.join(cmd)}")
    print(f"工作目录: {DET_ROOT}")
    subprocess.run(cmd, cwd=str(DET_ROOT), check=False)


if __name__ == "__main__":
    main()
