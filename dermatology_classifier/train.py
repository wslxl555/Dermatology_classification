"""
纯疾病分类模型训练入口。

用 labeled/ 与 unlabeled/ 中的图片（按目录派生疾病标签）训练三分类模型，
完全不涉及皮损检测与 .json 标注。

用法:
    python train.py                                        # 默认参数
    python train.py --image_size 384 --epochs 50
    python train.py --backbone resnet50 --no_pretrained
    python train.py --offline                              # 只用缓存权重
"""

import argparse
import json
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

from config import (
    DEFAULT_CONFIG, DISEASE_CLASSES,
    NUM_DISEASE_CLASSES, LABELED_ROOT, UNLABELED_ROOT,
    CHECKPOINT_DIR, OUTPUT_DIR,
)
from data import build_dataset, patient_stratified_split, DiseaseDataset, print_split_stats
from transforms import get_train_transforms, get_val_transforms
from model import build_model, BACKBONE_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="训练纯疾病三分类模型")
    p.add_argument("--labeled_root", type=str, default=str(LABELED_ROOT))
    p.add_argument("--unlabeled_root", type=str, default=str(UNLABELED_ROOT))
    p.add_argument("--backbone", type=str, default=DEFAULT_CONFIG["backbone"],
                   choices=BACKBONE_NAMES)
    p.add_argument("--no_pretrained", action="store_true",
                   help="禁用 ImageNet 预训练权重")
    p.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--backbone_lr", type=float, default=DEFAULT_CONFIG["backbone_lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--val_size", type=float, default=DEFAULT_CONFIG["val_size"])
    p.add_argument("--test_size", type=float, default=DEFAULT_CONFIG["test_size"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--augmentation", type=str,
                   default=DEFAULT_CONFIG["augmentation"],
                   choices=["medical", "standard", "none"])
    p.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--label_smoothing", type=float,
                   default=DEFAULT_CONFIG["label_smoothing"])
    p.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    p.add_argument("--patience", type=int,
                   default=DEFAULT_CONFIG["early_stopping_patience"])
    p.add_argument("--offline", action="store_true", help="只用缓存权重")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    image_ids = [b["image_id"] for b in batch]
    return {"image": images, "label": labels, "image_id": image_ids}


def compute_class_weights(samples, num_classes):
    counts = Counter(s["disease"] for s in samples)
    weights = torch.zeros(num_classes)
    for i, name in enumerate(DISEASE_CLASSES):
        weights[i] = 1.0 / max(counts.get(name, 0), 1)
    weights = weights / weights.sum() * num_classes
    return weights


def run_epoch(model, loader, device, criterion, optimizer=None):
    """训练（optimizer 非 None）或验证一个 epoch，返回 (avg_loss, labels, probs)。"""
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, n = 0.0, 0
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

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
    per_class_f1 = f1_score(labels, preds, average=None,
                            labels=list(range(n_classes)), zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    return {"acc": acc, "macro_f1": macro_f1,
            "per_class_f1": per_class_f1.tolist(),
            "auc": auc, "cm": cm}


def print_metrics(tag, m):
    print(f"  {tag:5s} | Acc {m['acc']:.4f} | Macro-F1 {m['macro_f1']:.4f} "
          f"| AUC {m['auc']:.4f}")
    print("    per-class F1: " + "  ".join(
        f"{DISEASE_CLASSES[i]}={m['per_class_f1'][i]:.4f}"
        for i in range(len(DISEASE_CLASSES))))


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 数据 ----
    print(f"Labeled root:   {args.labeled_root}")
    print(f"Unlabeled root: {args.unlabeled_root}")
    samples = build_dataset(args.labeled_root, args.unlabeled_root)
    print(f"合并去重后共 {len(samples)} 张图片")
    for name in DISEASE_CLASSES:
        print(f"  {name}: {sum(1 for s in samples if s['disease'] == name)} 张")

    splits = patient_stratified_split(
        samples, val_size=args.val_size, test_size=args.test_size, seed=args.seed,
    )
    print("\n数据划分:")
    print_split_stats(splits)

    # 保存划分，供 evaluate.py 复现同一测试集
    import pickle
    with open(OUTPUT_DIR / "splits.pkl", "wb") as f:
        pickle.dump(splits, f)

    train_tf = get_train_transforms(args.image_size, level=args.augmentation)
    val_tf = get_val_transforms(args.image_size)

    train_ds = DiseaseDataset(splits["train"], transform=train_tf)
    val_ds = DiseaseDataset(splits["val"], transform=val_tf)
    test_ds = DiseaseDataset(splits.get("test", []), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    # ---- 模型 ----
    model = build_model(
        backbone_name=args.backbone,
        num_classes=NUM_DISEASE_CLASSES,
        pretrained=not args.no_pretrained,
        offline=args.offline,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: {args.backbone} | 参数量 {n_params/1e6:.1f}M | 设备 {device}")

    # 差异化学习率：backbone 小 LR 微调，分类头大 LR 从头训练
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights = compute_class_weights(splits["train"], NUM_DISEASE_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=args.label_smoothing)
    print(f"类别权重: {[round(w, 3) for w in class_weights.tolist()]}")
    print(f"学习率: backbone={args.backbone_lr:.1e}  head={args.lr:.1e}  "
          f"label_smoothing={args.label_smoothing}")

    # ---- 训练 ----
    best_f1, best_epoch, bad_epochs = 0.0, 0, 0
    history = {"train": [], "val": []}

    print(f"\n{'='*60}\n开始训练\n{'='*60}")
    for epoch in range(args.epochs):
        t0 = time.time()

        tr_loss, tr_labels, tr_probs = run_epoch(
            model, train_loader, device, criterion, optimizer)
        scheduler.step()

        va_loss, va_labels, va_probs = run_epoch(model, val_loader, device, criterion)

        tr_m = compute_metrics(tr_labels, tr_probs)
        va_m = compute_metrics(va_labels, va_probs)
        history["train"].append({**tr_m, "loss": tr_loss, "cm": tr_m["cm"].tolist()})
        history["val"].append({**va_m, "loss": va_loss, "cm": va_m["cm"].tolist()})

        print(f"Epoch {epoch:3d} | {time.time()-t0:5.1f}s | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")
        print_metrics("Train", tr_m)
        print_metrics("Val", va_m)

        if va_m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = va_m["macro_f1"], epoch, 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_f1,
                "config": {
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "num_classes": NUM_DISEASE_CLASSES,
                    "class_names": DISEASE_CLASSES,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "seed": args.seed,
                },
            }
            torch.save(ckpt, CHECKPOINT_DIR / "best.pth")
            print(f"  >> 已保存最佳模型 (val macro-F1={best_f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  >> 早停：{args.patience} 个 epoch 未提升")
                break

    # ---- 保存历史 ----
    with open(OUTPUT_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n训练历史已保存: {OUTPUT_DIR / 'history.json'}")

    # ---- 测试集评估 ----
    print(f"\n{'='*60}")
    print(f"最佳 val macro-F1: {best_f1:.4f} @ epoch {best_epoch}")
    if splits.get("test"):
        ckpt = torch.load(CHECKPOINT_DIR / "best.pth", map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        _, te_labels, te_probs = run_epoch(model, test_loader, device, criterion)
        te_m = compute_metrics(te_labels, te_probs)
        print("测试集:")
        print_metrics("Test", te_m)
        print("\n混淆矩阵 (行=真实, 列=预测):")
        print("            " + " ".join(f"{n:>10}" for n in DISEASE_CLASSES))
        for i, row in enumerate(te_m["cm"]):
            print(f"  {DISEASE_CLASSES[i]:9s} " + " ".join(f"{v:10d}" for v in row))
    else:
        print("（未设置 test 划分，跳过测试集评估）")

    print(f"\n最佳模型已保存: {CHECKPOINT_DIR / 'best.pth'}")
    print("训练完成。")


if __name__ == "__main__":
    main()
