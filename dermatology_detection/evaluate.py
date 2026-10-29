"""
模型评估（基于 labeled/ 标注数据）。

用法:
    python evaluate.py --checkpoint best.pth
    python evaluate.py --checkpoint best.pth --split test
"""

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR, PROJECT_ROOT, DISEASE_CLASSES,
)
from data.prepare_labeled_data import scan_labeled_dataset, create_labeled_splits
from data.dataset import EczemaDataset
from data.transforms import get_val_transforms
from models.multi_task_model import build_model
from training.metrics import MetricsTracker


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint file to evaluate")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=None,
                        help="输入尺寸；默认取 checkpoint 里保存的 image_size")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device="cuda") -> dict:
    """运行评估并返回指标。"""
    model.eval()
    metrics = MetricsTracker(
        num_classes=len(DISEASE_CLASSES),
        class_names=DISEASE_CLASSES,
    )

    for batch in loader:
        images = batch["image"].to(device)
        disease_labels = batch["disease_label"]

        # 以推理模式前向传播,取出疾病分类的 logits 并转为 numpy
        output = model(images, targets=None)
        logits = output["disease_logits"].cpu().numpy()
        targets = disease_labels.numpy()

        metrics.update(logits, targets)

    return metrics.compute(), metrics.get_classification_report()


def main():
    """评估主入口:加载模型与数据,输出评估指标。"""
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载 checkpoint（先读，因为 image_size/backbone 需要从 config 取）
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    backbone = cfg.get("backbone", "convnext_tiny")
    detection_enabled = cfg.get("detection_enabled", False)
    detection_head = cfg.get("detection_head", "fcos")
    img_sz = cfg.get("image_size", (512, 512))
    img_sz = tuple(img_sz) if isinstance(img_sz, (tuple, list)) else (img_sz, img_sz)
    if args.image_size is not None:
        img_sz = (args.image_size, args.image_size)

    # 加载 labeled/ 数据划分（与训练共用同一套患者分层划分）
    splits_path = OUTPUT_DIR / "labeled_splits.pkl"
    if splits_path.exists():
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
    else:
        print("No saved labeled splits found. Scanning labeled dataset...")
        labeled_root = PROJECT_ROOT.parent / "labeled"
        samples = scan_labeled_dataset(labeled_root)
        splits = create_labeled_splits(
            samples, val_size=0.15, test_size=0.15, seed=42)

    eval_samples = splits.get(args.split, splits["val"])
    print(f"Evaluating on {args.split} split: {len(eval_samples)} samples "
          f"(backbone={backbone}, detection={detection_enabled}, image_size={img_sz})")

    # 数据变换与数据集
    transform = get_val_transforms(image_size=img_sz)
    dataset = EczemaDataset(eval_samples, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: {
            "image": torch.stack([item["image"] for item in batch]),
            "disease_label": torch.tensor([item["disease_label"] for item in batch]),
            "image_id": [item["image_id"] for item in batch],
        },
    )

    # 加载模型
    model = build_model(
        backbone_name=backbone,
        detection_enabled=detection_enabled,
        detection_head=detection_head,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 评估
    metrics, report = evaluate(model, loader, device)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy:      {metrics.get('accuracy', 0):.4f}")
    print(f"Macro F1:      {metrics.get('macro_f1', 0):.4f}")
    print(f"Weighted F1:   {metrics.get('weighted_f1', 0):.4f}")
    print(f"Macro Precision: {metrics.get('macro_precision', 0):.4f}")
    print(f"Macro Recall:  {metrics.get('macro_recall', 0):.4f}")

    auc = metrics.get("auc_roc_ovr", float("nan"))
    if not (auc != auc):  # 非 NaN
        print(f"AUC-ROC (OvR): {auc:.4f}")

    print(f"\nPer-class F1:")
    for name in DISEASE_CLASSES:
        f1 = metrics.get(f"f1_{name}", 0)
        print(f"  {name:25s}: {f1:.4f}")

    print(f"\nConfusion Matrix:")
    cm = metrics.get("confusion_matrix", [])
    print("               " + " ".join(f"{n:>6}" for n in DISEASE_CLASSES))
    for i, row in enumerate(cm):
        print(f"  {DISEASE_CLASSES[i]:12s} " + " ".join(f"{v:6d}" for v in row))

    print(f"\n{report}")


if __name__ == "__main__":
    main()
