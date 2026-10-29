"""
顺序训练三个配置并各存一份带标签的 checkpoint，用于对比：

  1. baseline:   convnext_tiny + fcos
  2. backbone:   swin_t        + fcos
  3. detector:   convnext_tiny + faster_rcnn

三者使用完全相同的训练设置（256px / batch2 / 相同 epochs / 相同 seed），
保证对比公平。训练结束后把 best.pth 复制为 compare_<tag>_best.pth。

用法:
    python run_comparison.py --epochs 5
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent
CHECKPOINT_DIR = _project_root / "outputs" / "checkpoints"

RUNS = [
    {"tag": "baseline_convnext_fcos", "backbone": "convnext_tiny", "head": "fcos"},
    {"tag": "backbone_swin_fcos", "backbone": "swin_t", "head": "fcos"},
    {"tag": "detector_convnext_fasterrcnn", "backbone": "convnext_tiny", "head": "faster_rcnn"},
]


def run_one(tag, backbone, head, epochs):
    print(f"\n{'#'*70}\n# RUN: {tag}  (backbone={backbone}, head={head}, epochs={epochs})\n{'#'*70}\n")
    cmd = [
        sys.executable, "train_labeled.py",
        "--backbone", backbone,
        "--detection_head", head,
        "--epochs", str(epochs),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_project_root))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL] {tag} exited with code {proc.returncode} after {elapsed:.0f}s")
        return False

    src = CHECKPOINT_DIR / "best.pth"
    dst = CHECKPOINT_DIR / f"compare_{tag}_best.pth"
    if src.exists():
        shutil.copy(src, dst)
        print(f"[OK] {tag} done in {elapsed:.0f}s; saved {dst.name}")
    else:
        print(f"[WARN] {tag}: best.pth not found (may have early-stopped before epoch 0)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated tags to run (default: all)")
    args = parser.parse_args()

    # 备份现有 baseline（epoch 2）以备不时之需
    cur_best = CHECKPOINT_DIR / "best.pth"
    if cur_best.exists():
        shutil.copy(cur_best, CHECKPOINT_DIR / "compare_baseline_ep2_best.pth")

    selected = args.runs.split(",") if args.runs else None
    for r in RUNS:
        if selected and r["tag"] not in selected:
            continue
        run_one(r["tag"], r["backbone"], r["head"], args.epochs)

    print("\nComparison runs complete.")


if __name__ == "__main__":
    main()
