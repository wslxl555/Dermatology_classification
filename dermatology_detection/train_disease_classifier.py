"""
纯疾病分类模型训练入口。

与多任务检测模型不同，本模型只做疾病三分类（湿疹 / 大疱型类天疱疮 / 荨麻疹）。
训练与验证都**只读取图片 + 目录派生的疾病标签**，完全不读取 .json 标注文件。
（疾病标签来自图片所在目录，而非 LabelMe 标注。）

用法:
    python train_disease_classifier.py                          # convnext_tiny, 224px, 20 epoch
    python train_disease_classifier.py --image_size 320 --epochs 30
    python train_disease_classifier.py --backbone resnet50 --no_pretrained
    python train_disease_classifier.py --offline                # 只用缓存权重
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import DISEASE_CLASSES, DISEASE_CLASSES_ZH, OUTPUT_DIR, CHECKPOINT_DIR
from data.prepare_labeled_data import scan_labeled_dataset, create_labeled_splits
from data.disease_cls_dataset import DiseaseClassificationDataset
from data.transforms import get_cls_train_transforms, get_cls_val_transforms
from models.classification_model import build_classification_model


def parse_args():
    p = argparse.ArgumentParser(description="Train a pure disease classification model")
    p.add_argument("--labeled_root", type=str, default=None,
                   help="Path to labeled/ folder (default: ../labeled)")
    p.add_argument("--backbone", type=str, default="convnext_tiny",
                   choices=["convnext_tiny", "convnext_small", "resnet50", "swin_t"])
    p.add_argument("--no_pretrained", action="store_true",
                   help="Disable ImageNet pretrained weights")
    p.add_argument("--image_size", type=int, default=224, help="Square input size")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4, help="Backbone LR; head uses 10x")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augmentation", type=str, default="medical",
                   choices=["medical", "standard", "none"])
    p.add_argument("--offline", action="store_true", help="Use cached weights only")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["disease_label"] for b in batch], dtype=torch.long)
    image_ids = [b["image_id"] for b in batch]
    return {"image": images, "disease_label": labels, "image_id": image_ids}


def compute_class_weights(samples, num_classes):
    """按类别样本数的倒数计算平衡权重，并归一化使均值为 1。"""
    counts = Counter(s["disease"] for s in samples)
    weights = torch.zeros(num_classes)
    for i, name in enumerate(DISEASE_CLASSES):
        weights[i] = 1.0 / max(counts.get(name, 0), 1)
    weights = weights / weights.sum() * num_classes
    return weights


def run_epoch(model, loader, device, criterion, optimizer=None):
    """训练（optimizer 非 None）或验证一个 epoch。

    返回 (avg_loss, labels, probs)。
    """
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    n = 0
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["disease_label"].to(device)

        if train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return total_loss / max(n, 1), all_labels, all_probs


def compute_metrics(labels, probs):
    preds = probs.argmax(axis=1)
    n_classes = len(DISEASE_CLASSES)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(
        labels, preds, average=None,
        labels=list(range(n_classes)), zero_division=0,
    )
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    return {"acc": acc, "macro_f1": macro_f1, "per_class_f1": per_class,
            "auc": auc, "cm": cm}


def _print_metrics(tag, m):
    print(f"  {tag:5s} | Acc: {m['acc']:.4f} | Macro F1: {m['macro_f1']:.4f} "
          f"| AUC: {m['auc']:.4f}")
    row = "    per-class F1: " + "  ".join(
        f"{DISEASE_CLASSES[i]}={m['per_class_f1'][i]:.4f}"
        for i in range(len(DISEASE_CLASSES))
    )
    print(row)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    labeled_root = (Path(args.labeled_root)
                    if args.labeled_root else _project_root.parent / "labeled")
    print(f"Labeled root: {labeled_root}")

    # 数据划分（按患者分层）
    splits_path = OUTPUT_DIR / "disease_cls_splits.pkl"
    if splits_path.exists():
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        print(f"Loaded splits from {splits_path}")
    else:
        samples = scan_labeled_dataset(labeled_root)
        splits = create_labeled_splits(
            samples, val_size=args.val_size, test_size=args.test_size, seed=args.seed,
        )
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        print(f"Splits saved to {splits_path}")

    train_samples, val_samples = splits["train"], splits["val"]
    test_samples = splits.get("test", [])
    print(f"\nTrain {len(train_samples)} | Val {len(val_samples)} | Test {len(test_samples)}")
    for name, ss in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        print(f"  {name}: {dict(Counter(s['disease'] for s in ss))}")

    # 变换与数据集（纯分类，不涉及 .json）
    train_tf = get_cls_train_transforms((args.image_size, args.image_size), level=args.augmentation)
    val_tf = get_cls_val_transforms((args.image_size, args.image_size))

    train_ds = DiseaseClassificationDataset(train_samples, transform=train_tf)
    val_ds = DiseaseClassificationDataset(val_samples, transform=val_tf)
    test_ds = DiseaseClassificationDataset(test_samples, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    # 模型
    model = build_classification_model(
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        offline=args.offline,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.backbone} | params: {n_params/1e6:.1f}M | device: {device}")

    # 差异化学习率：backbone 小 LR 微调，分类头大 LR 从头训练
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights = compute_class_weights(train_samples, len(DISEASE_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    # 训练循环
    best_f1 = 0.0
    best_epoch = 0
    history = {"train": [], "val": []}

    print(f"\n{'='*60}")
    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, tr_labels, tr_probs = run_epoch(
            model, train_loader, device, criterion, optimizer)
        scheduler.step()

        val_loss, va_labels, va_probs = run_epoch(model, val_loader, device, criterion)

        tr_m = compute_metrics(tr_labels, tr_probs)
        va_m = compute_metrics(va_labels, va_probs)
        history["train"].append({**tr_m, "loss": train_loss, "cm": tr_m["cm"].tolist()})
        history["val"].append({**va_m, "loss": val_loss, "cm": va_m["cm"].tolist()})

        print(f"Epoch {epoch:3d} | {time.time()-t0:5.1f}s | LR {scheduler.get_last_lr()[0]:.2e}")
        _print_metrics("Train", tr_m)
        _print_metrics("Val", va_m)

        if va_m["macro_f1"] > best_f1:
            best_f1 = va_m["macro_f1"]
            best_epoch = epoch
            ckpt_path = CHECKPOINT_DIR / "disease_cls_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_f1,
                "config": {
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "num_classes": len(DISEASE_CLASSES),
                    "class_names": DISEASE_CLASSES,
                },
            }, ckpt_path)
            print(f"  >> Checkpoint saved: {ckpt_path}")

    # 保存训练历史
    history_path = OUTPUT_DIR / "disease_cls_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str, ensure_ascii=False)

    # 测试集评估（最佳 checkpoint）
    print(f"\n{'='*60}")
    print(f"Best val F1: {best_f1:.4f} at epoch {best_epoch}")
    ckpt = torch.load(CHECKPOINT_DIR / "disease_cls_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, te_labels, te_probs = run_epoch(model, test_loader, device, criterion)
    te_m = compute_metrics(te_labels, te_probs)
    print("Test set:")
    _print_metrics("Test", te_m)

    print(f"\nHistory saved to: {history_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
