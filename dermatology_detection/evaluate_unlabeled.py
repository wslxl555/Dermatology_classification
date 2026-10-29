"""
在 unlabeled/ 文件夹上评估训练好的 dermatology_detection 多任务模型。

unlabeled/ 按疾病分目录（湿疹 / 类天疱疹 / 荨麻疹），目录名即疾病真值。
unlabeled 图片没有 .json 标注，因此这里只评估疾病三分类（无法评估皮损检测分支）。

用法:
    python evaluate_unlabeled.py --checkpoint best.pth
    python evaluate_unlabeled.py --checkpoint best.pth --unlabeled_root ../unlabeled
    python evaluate_unlabeled.py --checkpoint best.pth --batch_size 16
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR, DISEASE_CLASSES, DISEASE_CLASSES_ZH,
)
from data.prepare_labeled_data import scan_labeled_dataset
from data.disease_cls_dataset import DiseaseClassificationDataset
from data.transforms import get_cls_val_transforms
from models.multi_task_model import build_model
from training.metrics import MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate disease classification on unlabeled/ test set"
    )
    parser.add_argument("--checkpoint", type=str, default="best.pth",
                        help="Checkpoint file name (under outputs/checkpoints/)")
    parser.add_argument("--unlabeled_root", type=str, default=None,
                        help="Path to unlabeled/ folder (default: ../unlabeled)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override input size (default: from checkpoint config)")
    return parser.parse_args()


def collate_fn(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["disease_label"] for b in batch], dtype=torch.long)
    image_ids = [b["image_id"] for b in batch]
    return {"image": images, "disease_label": labels, "image_id": image_ids}


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, str, list[dict]]:
    """返回 (metrics, classification_report, per_image_predictions)。"""
    model.eval()
    metrics = MetricsTracker(
        num_classes=len(DISEASE_CLASSES), class_names=DISEASE_CLASSES,
    )
    per_image = []

    for batch in loader:
        images = batch["image"].to(device)
        disease_labels = batch["disease_label"]

        output = model(images, targets=None)
        logits = output["disease_logits"].cpu().numpy()
        probs = torch.softmax(output["disease_logits"], dim=-1).cpu().numpy()

        preds = probs.argmax(axis=1)
        for i, image_id in enumerate(batch["image_id"]):
            per_image.append({
                "image_id": image_id,
                "true": int(disease_labels[i].item()),
                "true_name": DISEASE_CLASSES[int(disease_labels[i].item())],
                "pred": int(preds[i]),
                "pred_name": DISEASE_CLASSES[int(preds[i])],
                "confidence": float(probs[i][preds[i]]),
                "probs": {DISEASE_CLASSES[c]: float(probs[i][c])
                          for c in range(len(DISEASE_CLASSES))},
            })

        metrics.update(logits, disease_labels.numpy())

    return metrics.compute(), metrics.get_classification_report(), per_image


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 检查 checkpoint 路径（支持纯文件名或绝对路径）
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = CHECKPOINT_DIR / args.checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    backbone = cfg.get("backbone", "convnext_tiny")
    detection_enabled = cfg.get("detection_enabled", False)
    detection_head = cfg.get("detection_head", "fcos")
    img_sz = cfg.get("image_size", (512, 512))
    img_sz = tuple(img_sz) if isinstance(img_sz, (tuple, list)) else (img_sz, img_sz)
    if args.image_size is not None:
        img_sz = (args.image_size, args.image_size)

    # 扫描 unlabeled/ 测试集（目录名 → 疾病真值，无 json 标注）
    unlabeled_root = args.unlabeled_root
    if unlabeled_root is None:
        unlabeled_root = _project_root.parent / "unlabeled"
    unlabeled_root = Path(unlabeled_root).resolve()
    print(f"Unlabeled root: {unlabeled_root}")

    samples = scan_labeled_dataset(unlabeled_root)
    if not samples:
        raise RuntimeError(f"No images found under {unlabeled_root}")

    counts = Counter(s["disease"] for s in samples)
    print(f"Test samples: {len(samples)} | {dict(counts)}")

    # 数据集与加载器
    transform = get_cls_val_transforms(img_sz)
    dataset = DiseaseClassificationDataset(samples, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # 构建模型并加载权重（pretrained=False：checkpoint 会覆盖所有权重，
    # 避免在内存受限的机器上再额外加载一份 ImageNet 预训练权重）。
    model = build_model(
        backbone_name=backbone,
        pretrained=False,
        detection_enabled=detection_enabled,
        detection_head=detection_head,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Model loaded: backbone={backbone}, detection={detection_enabled}, "
          f"image_size={img_sz}")

    # 评估
    metrics, report, per_image = evaluate(model, loader, device)

    # 打印结果
    print("\n" + "=" * 60)
    print("EVALUATION ON unlabeled/ TEST SET")
    print("=" * 60)
    print(f"\nTest samples: {len(samples)}")
    print(f"Accuracy:        {metrics.get('accuracy', 0):.4f}")
    print(f"Macro F1:        {metrics.get('macro_f1', 0):.4f}")
    print(f"Weighted F1:     {metrics.get('weighted_f1', 0):.4f}")
    print(f"Macro Precision: {metrics.get('macro_precision', 0):.4f}")
    print(f"Macro Recall:    {metrics.get('macro_recall', 0):.4f}")

    auc = metrics.get("auc_roc_ovr", float("nan"))
    if auc == auc:  # 非 NaN
        print(f"AUC-ROC (OvR):   {auc:.4f}")

    print(f"\nPer-class F1:")
    for name in DISEASE_CLASSES:
        f1 = metrics.get(f"f1_{name}", 0)
        zh = DISEASE_CLASSES_ZH.get(name, name)
        print(f"  {name:25s} ({zh}): {f1:.4f}")

    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    cm = metrics.get("confusion_matrix", [])
    print("               " + " ".join(f"{n:>8}" for n in DISEASE_CLASSES))
    for i, row in enumerate(cm):
        print(f"  {DISEASE_CLASSES[i]:12s} " + " ".join(f"{v:8d}" for v in row))

    print(f"\n{report}")

    # 保存报告
    report_path = OUTPUT_DIR / "evaluation_unlabeled.json"
    serializable = {
        "checkpoint": str(ckpt_path),
        "backbone": backbone,
        "detection_enabled": detection_enabled,
        "image_size": list(img_sz),
        "num_test_samples": len(samples),
        "test_distribution": {k: v for k, v in counts.items()},
        "metrics": {k: (v if not isinstance(v, list) else v)
                    for k, v in metrics.items()},
        "confusion_matrix": cm,
    }
    # 将 numpy 类型转为原生类型
    def _to_native(o):
        if isinstance(o, dict):
            return {k: _to_native(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_native(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, float) and o != o:  # NaN
            return None
        return o

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_to_native(serializable), f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")

    # 保存逐图预测 CSV
    csv_path = OUTPUT_DIR / "evaluation_unlabeled_predictions.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "true", "pred", "true_name", "pred_name",
                         "confidence"])
        for p in per_image:
            writer.writerow([p["image_id"], p["true"], p["pred"],
                             p["true_name"], p["pred_name"], p["confidence"]])
    print(f"Per-image predictions saved to: {csv_path}")


if __name__ == "__main__":
    main()
