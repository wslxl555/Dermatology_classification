"""
患者级 k-fold 交叉验证。

按 (disease, patient_id) 分组，用 StratifiedKFold 做 k 折，
同一患者的所有图片只出现在同一折。每折在 k-1 折上训练、在 1 折上验证，
报告图像级 + 患者级指标，最后给出 k 折均值 ± 标准差。

用法:
    python cross_validate.py --backbone swin_t --epochs 15 --folds 5
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from config import (
    DEFAULT_CONFIG, NUM_DISEASE_CLASSES, LABELED_ROOT, UNLABELED_ROOT,
    DISEASE_TO_IDX,
)
from data import build_dataset, DiseaseDataset
from transforms import get_train_transforms, get_val_transforms
from model import build_model, BACKBONE_NAMES
from train import run_epoch, compute_metrics, compute_class_weights
from evaluate import patient_level_metrics


def parse_args():
    p = argparse.ArgumentParser(description="患者级 k-fold 交叉验证")
    p.add_argument("--labeled_root", type=str, default=str(LABELED_ROOT))
    p.add_argument("--unlabeled_root", type=str, default=str(UNLABELED_ROOT))
    p.add_argument("--backbone", type=str, default=DEFAULT_CONFIG["backbone"],
                   choices=BACKBONE_NAMES)
    p.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--backbone_lr", type=float, default=DEFAULT_CONFIG["backbone_lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    p.add_argument("--label_smoothing", type=float,
                   default=DEFAULT_CONFIG["label_smoothing"])
    p.add_argument("--augmentation", type=str, default=DEFAULT_CONFIG["augmentation"],
                   choices=["medical", "standard", "none"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--offline", action="store_true", help="只用缓存权重")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def cv_collate(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    image_ids = [b["image_id"] for b in batch]
    patient_ids = [b["patient_id"] for b in batch]
    return {"image": images, "label": labels, "image_id": image_ids,
            "patient_id": patient_ids}


def make_folds(samples, n_splits, seed):
    """按 (disease, patient_id) 分组做患者级分层 k 折。"""
    groups = defaultdict(list)
    for s in samples:
        groups[(s["disease"], s["patient_id"])].append(s)
    keys = list(groups.keys())
    y = [DISEASE_TO_IDX[k[0]] for k in keys]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, va_idx in skf.split(keys, y):
        va_keys = {keys[i] for i in va_idx}
        train = [s for s in samples if (s["disease"], s["patient_id"]) not in va_keys]
        val = [s for s in samples if (s["disease"], s["patient_id"]) in va_keys]
        folds.append((train, val))
    return folds


@torch.no_grad()
def cv_validate(model, loader, device):
    model.eval()
    all_labels, all_probs, all_pids = [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())
        all_pids.extend(batch["patient_id"])
    return (np.concatenate(all_labels),
            np.concatenate(all_probs, axis=0),
            all_pids)


def train_fold(args, device, train_samples, val_samples):
    train_tf = get_train_transforms(args.image_size, level=args.augmentation)
    val_tf = get_val_transforms(args.image_size)
    train_loader = DataLoader(DiseaseDataset(train_samples, transform=train_tf),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=cv_collate)
    val_loader = DataLoader(DiseaseDataset(val_samples, transform=val_tf),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=cv_collate)

    model = build_model(
        backbone_name=args.backbone, num_classes=NUM_DISEASE_CLASSES,
        pretrained=True, offline=args.offline,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)

    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights = compute_class_weights(train_samples, NUM_DISEASE_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=args.label_smoothing)

    best_f1 = -1.0
    best_state = None
    for epoch in range(args.epochs):
        run_epoch(model, train_loader, device, criterion, optimizer)
        scheduler.step()
        va_labels, va_probs, va_pids = cv_validate(model, val_loader, device)
        va_f1 = compute_metrics(va_labels, va_probs)["macro_f1"]
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = (va_labels, va_probs, va_pids)
    return best_f1, best_state


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples = build_dataset(args.labeled_root, args.unlabeled_root)
    print(f"合并去重后共 {len(samples)} 张图片")
    folds = make_folds(samples, args.folds, args.seed)

    print(f"\n{'='*60}\n患者级 {args.folds}-fold 交叉验证\n{'='*60}")
    print(f"backbone={args.backbone}  epochs={args.epochs}  "
          f"backbone_lr={args.backbone_lr:.1e}  head_lr={args.lr:.1e}  "
          f"dropout={args.dropout}  weight_decay={args.weight_decay:.1e}")

    img_accs, img_f1s, img_aucs = [], [], []
    pat_accs, pat_f1s, pat_allc, pat_aucs = [], [], [], []

    for fold_idx, (tr, va) in enumerate(folds):
        print(f"\n--- Fold {fold_idx+1}/{args.folds}: "
              f"train={len(tr)} 张, val={len(va)} 张 ---")
        best_f1, best_state = train_fold(args, device, tr, va)
        va_labels, va_probs, va_pids = best_state
        preds = va_probs.argmax(axis=1)

        img_m = compute_metrics(va_labels, va_probs)
        pat_m = patient_level_metrics(va_pids, va_labels, preds, probs=va_probs)

        img_accs.append(img_m["acc"])
        img_f1s.append(img_m["macro_f1"])
        img_aucs.append(img_m["auc"])
        pat_accs.append(pat_m["accuracy"])
        pat_f1s.append(pat_m["macro_f1"])
        pat_allc.append(pat_m["all_correct_accuracy"])
        pat_aucs.append(pat_m["auc"])

        print(f"  best val macro-F1: {best_f1:.4f}")
        print(f"  图像级: Acc {img_m['acc']:.4f} | Macro-F1 {img_m['macro_f1']:.4f} "
              f"| AUC {img_m['auc']:.4f}")
        print(f"  患者级: Acc {pat_m['accuracy']:.4f} | Macro-F1 {pat_m['macro_f1']:.4f} "
              f"| AUC {pat_m['auc']:.4f} | 全对占比 {pat_m['all_correct_accuracy']:.4f}")

    def mean_std(xs):
        xs = np.asarray(xs, dtype=float)
        return xs.mean(), xs.std()

    print(f"\n{'='*60}\n{args.folds}-fold 汇总 (mean ± std)\n{'='*60}")
    rows = [
        ("图像级 Accuracy", img_accs), ("图像级 Macro-F1", img_f1s),
        ("图像级 AUC", img_aucs), ("患者级 Accuracy", pat_accs),
        ("患者级 Macro-F1", pat_f1s), ("患者级 AUC", pat_aucs),
        ("患者级 全对占比", pat_allc),
    ]
    for name, xs in rows:
        m, s = mean_std(xs)
        print(f"  {name:16s}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
