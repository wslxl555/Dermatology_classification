"""
训练"皮损标注统计信息 → 疾病"分类器（两阶段流程的第二阶段）。

第一阶段检测模型对图片自动标注（box + annotation）；
本脚本学习"各类 annotation 的统计信息"与"各类疾病之间的关系"，
以一张图片的皮损类型统计直方图（15 维）为输入，输出 3 类疾病。

训练完成后会报告：
  1. 经验统计：训练集中各疾病的皮损类型分布（annotation 与疾病的关系）；
  2. 学习到的"皮损类型 → 疾病"关联矩阵；
  3. 分类指标（Acc / Macro-F1 / 各类 F1 / AUC / 混淆矩阵）。

用法:
    python train_annotation_classifier.py                    # 默认占比特征
    python train_annotation_classifier.py --feature multi_hot
    python train_annotation_classifier.py --feature count
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    DISEASE_CLASSES, DISEASE_CLASSES_ZH,
    LESION_CLASSES, LESION_CLASSES_ZH,
    OUTPUT_DIR, CHECKPOINT_DIR,
)
from data.prepare_labeled_data import (
    scan_labeled_dataset, create_labeled_splits, DISEASE_TO_IDX,
)
from data.annotation_features import load_lesion_histogram, histogram_to_features
from models.annotation_stat_classifier import build_annotation_classifier


def parse_args():
    p = argparse.ArgumentParser(
        description="Train annotation-statistics disease classifier")
    p.add_argument("--labeled_root", type=str, default=None,
                   help="Path to labeled/ folder (default: ../labeled)")
    p.add_argument("--feature", type=str, default="proportion",
                   choices=["proportion", "count", "multi_hot"],
                   help="Which histogram feature to use as input")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def build_xy(samples, feature_key):
    """把样本列表转成 (X[N,15], y[N]) 张量。"""
    X, y = [], []
    for s in samples:
        counts = load_lesion_histogram(s.get("json_path"))
        feats = histogram_to_features(counts)
        X.append(feats[feature_key])
        y.append(DISEASE_TO_IDX[s["disease"]])
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def compute_class_weights(y, num_classes):
    counts = Counter(y.tolist())
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        weights[i] = 1.0 / max(counts.get(i, 0), 1)
    weights = weights / weights.sum() * num_classes
    return weights


def compute_metrics(labels, probs):
    preds = probs.argmax(axis=1)
    n = len(DISEASE_CLASSES)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None,
                         labels=list(range(n)), zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
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


def report_associations(model, train_samples, feature_key):
    """
    报告（并返回）经验统计 + 学习到的"皮损类型 → 疾病"关联。
    """
    # 1) 经验统计：各疾病的皮损类型平均占比（来自训练集 .json 标注）
    emp = np.zeros((len(DISEASE_CLASSES), len(LESION_CLASSES)), dtype=np.float32)
    counts_by_disease = Counter()
    for s in train_samples:
        d = DISEASE_TO_IDX[s["disease"]]
        feats = histogram_to_features(load_lesion_histogram(s.get("json_path")))
        emp[d] += feats[feature_key]
        counts_by_disease[d] += 1
    for d in range(len(DISEASE_CLASSES)):
        n = counts_by_disease[d]
        if n > 0:
            emp[d] /= n

    # 2) 学习到的关联矩阵（softmax 归一化到各疾病）
    with torch.no_grad():
        assoc = model.association_matrix().cpu().numpy()          # [15, 3]
    assoc_norm = np.exp(assoc - assoc.max(axis=1, keepdims=True))
    assoc_norm /= assoc_norm.sum(axis=1, keepdims=True)           # [15, 3] 行和为 1

    return emp, assoc_norm


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    labeled_root = (Path(args.labeled_root)
                    if args.labeled_root else _project_root.parent / "labeled")
    print(f"Labeled root: {labeled_root}")
    print(f"Feature: {args.feature}")

    # 复用检测流程的分裂（同一套患者分层），保证两阶段一致。
    splits_path = OUTPUT_DIR / "labeled_splits.pkl"
    if splits_path.exists():
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        print(f"Loaded splits from {splits_path}")
    else:
        samples = scan_labeled_dataset(labeled_root)
        splits = create_labeled_splits(samples, val_size=0.15, test_size=0.15,
                                       seed=args.seed)
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        print(f"Splits saved to {splits_path}")

    train_samples, val_samples = splits["train"], splits["val"]
    test_samples = splits.get("test", [])
    print(f"\nTrain {len(train_samples)} | Val {len(val_samples)} "
          f"| Test {len(test_samples)}")

    X_tr, y_tr = build_xy(train_samples, args.feature)
    X_va, y_va = build_xy(val_samples, args.feature)
    X_te, y_te = build_xy(test_samples, args.feature)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_va, y_va)
    test_ds = TensorDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_annotation_classifier(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: annotation-stat classifier | params: {n_params} | device: {device}")

    class_weights = compute_class_weights(y_tr, len(DISEASE_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    print(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    def run_epoch(loader, train=False):
        model.train() if train else model.eval()
        total, n = 0.0, 0
        all_labels, all_probs = [], []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item() * yb.size(0)
            n += yb.size(0)
            all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())
        return (total / max(n, 1),
                np.concatenate(all_labels), np.concatenate(all_probs))

    best_f1, best_epoch, best_state = 0.0, 0, None
    print(f"\n{'='*60}")
    for epoch in range(args.epochs):
        _, tr_l, tr_p = run_epoch(train_loader, train=True)
        _, va_l, va_p = run_epoch(val_loader, train=False)
        va_m = compute_metrics(va_l, va_p)
        if va_m["macro_f1"] > best_f1:
            best_f1 = va_m["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            tr_m = compute_metrics(tr_l, tr_p)
            print(f"Epoch {epoch:3d} | Train F1 {tr_m['macro_f1']:.4f} "
                  f"| Val F1 {va_m['macro_f1']:.4f}")

    model.load_state_dict(best_state)
    ckpt_path = CHECKPOINT_DIR / "annotation_cls_best.pth"
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "best_val_f1": best_f1,
        "config": {
            "feature": args.feature,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "num_lesion_classes": len(LESION_CLASSES),
            "num_diseases": len(DISEASE_CLASSES),
        },
    }, ckpt_path)
    print(f"\nBest val F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved: {ckpt_path}")

    # 测试集评估
    _, te_l, te_p = run_epoch(test_loader, train=False)
    te_m = compute_metrics(te_l, te_p)
    print(f"\nTest set ({args.feature}):")
    _print_metrics("Test", te_m)

    # 报告"皮损类型 → 疾病"的关系
    emp, assoc = report_associations(model, train_samples, args.feature)
    print(f"\n{'='*60}")
    print("ANNOTATION STATISTICS → DISEASE RELATIONSHIPS")
    print("(行 = 皮损类型；列 = 疾病。数值为归一化关联强度，越大越具判别性)")
    print(f"{'='*60}")
    header = "lesion".ljust(16) + "".join(f"{DISEASE_CLASSES[i]:>20}" for i in range(3))
    print(header)
    for li, lesion in enumerate(LESION_CLASSES):
        row = LESION_CLASSES_ZH[lesion].ljust(16)
        row += "".join(f"{assoc[li, d]:>20.3f}" for d in range(3))
        print(row)

    # 每个皮损类型最相关的疾病（经验统计）
    print(f"\nMost-associated disease per lesion type (empirical, {args.feature}):")
    for li, lesion in enumerate(LESION_CLASSES):
        d = int(np.argmax(emp[:, li]))
        print(f"  {LESION_CLASSES_ZH[lesion]} ({lesion}): "
              f"{DISEASE_CLASSES_ZH[DISEASE_CLASSES[d]]} "
              f"(emp {emp[d, li]:.3f}, learned {assoc[li, d]:.3f})")

    # 保存报告
    report = {
        "feature": args.feature,
        "test_metrics": {k: (v.tolist() if hasattr(v, "tolist") else v)
                         for k, v in te_m.items()},
        "association_matrix": assoc.tolist(),
        "empirical_distribution": emp.tolist(),
        "lesion_classes": LESION_CLASSES,
        "disease_classes": DISEASE_CLASSES,
    }
    report_path = OUTPUT_DIR / "annotation_stat_association.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nAssociation report saved: {report_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
