"""
评估训练好的纯疾病分类模型（在 held-out 测试集上）。

复用 train.py 保存的 outputs/splits.pkl（若存在），否则按相同 seed 重新划分。

用法:
    python evaluate.py                          # 用 outputs/checkpoints/best.pth
    python evaluate.py --checkpoint best.pth
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)

from config import (
    DEFAULT_CONFIG, DISEASE_CLASSES, DISEASE_CLASSES_ZH,
    NUM_DISEASE_CLASSES, LABELED_ROOT, UNLABELED_ROOT,
    CHECKPOINT_DIR, OUTPUT_DIR,
)
from data import build_dataset, patient_stratified_split, DiseaseDataset
from transforms import get_val_transforms
from model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="评估纯疾病分类模型")
    p.add_argument("--checkpoint", type=str, default="best.pth")
    p.add_argument("--labeled_root", type=str, default=str(LABELED_ROOT))
    p.add_argument("--unlabeled_root", type=str, default=str(UNLABELED_ROOT))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    image_ids = [b["image_id"] for b in batch]
    patient_ids = [b["patient_id"] for b in batch]
    return {"image": images, "label": labels, "image_id": image_ids,
            "patient_id": patient_ids}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs, all_ids, all_pids = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())
        all_ids.extend(batch["image_id"])
        all_pids.extend(batch["patient_id"])
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return labels, probs, all_ids, all_pids


def patient_level_metrics(patient_ids, labels, preds, probs=None):
    """患者级评估：同一患者多图先聚合，再计算患者级指标。

    硬预测多数投票；若传入 probs（N×C），患者级概率取均值，用于计算患者级 AUC。

    返回 dict：n_patients / accuracy / macro_f1 / per_class_f1 / confusion_matrix /
    all_correct_accuracy / all_correct_count / auc。
    """
    groups = defaultdict(lambda: {"true": None, "preds": [], "probs": []})
    for pid, t, p in zip(patient_ids, labels, preds):
        g = groups[(pid, int(t))]
        g["true"] = int(t)
        g["preds"].append(int(p))
    if probs is not None:
        for (pid, t), pr in zip(zip(patient_ids, labels), probs):
            groups[(pid, int(t))]["probs"].append(pr)

    p_true, p_pred, p_probs = [], [], []
    all_correct = 0
    n_patients = len(groups)
    for g in groups.values():
        ps = g["preds"]
        maj = max(set(ps), key=ps.count)
        p_true.append(g["true"])
        p_pred.append(maj)
        if all(p == g["true"] for p in ps):
            all_correct += 1
        if g["probs"]:
            p_probs.append(np.mean(g["probs"], axis=0))

    p_true = np.asarray(p_true)
    p_pred = np.asarray(p_pred)
    acc = accuracy_score(p_true, p_pred)
    macro_f1 = f1_score(p_true, p_pred, average="macro", zero_division=0)
    per_class = f1_score(p_true, p_pred, average=None,
                         labels=list(range(NUM_DISEASE_CLASSES)), zero_division=0)
    cm = confusion_matrix(p_true, p_pred, labels=list(range(NUM_DISEASE_CLASSES)))

    auc = float("nan")
    if probs is not None and len(p_probs) == n_patients:
        try:
            auc = roc_auc_score(p_true, np.stack(p_probs),
                                multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")

    return {
        "n_patients": n_patients,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class,
        "confusion_matrix": cm,
        "all_correct_accuracy": all_correct / max(n_patients, 1),
        "all_correct_count": all_correct,
        "auc": auc,
    }


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = CHECKPOINT_DIR / args.checkpoint
    print(f"加载模型: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    image_size = cfg.get("image_size", 224)
    backbone = cfg.get("backbone", "convnext_tiny")

    # 重建数据 + 划分
    samples = build_dataset(args.labeled_root, args.unlabeled_root)
    splits_path = OUTPUT_DIR / "splits.pkl"
    if splits_path.exists():
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        print(f"复用划分: {splits_path}")
    else:
        seed = cfg.get("seed", DEFAULT_CONFIG["seed"])
        splits = patient_stratified_split(
            samples, val_size=DEFAULT_CONFIG["val_size"],
            test_size=DEFAULT_CONFIG["test_size"], seed=seed,
        )
        print(f"未找到划分文件，用 seed={seed} 重新划分")

    test_samples = splits.get("test", [])
    if not test_samples:
        # 无 test 时退回到 val 作为评估集
        test_samples = splits.get("val", [])
        print("[提示] 无 test 划分，改用 val 划分评估")
    print(f"评估样本数: {len(test_samples)}")

    tf = get_val_transforms(image_size)
    ds = DiseaseDataset(test_samples, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    model = build_model(
        backbone_name=backbone, num_classes=NUM_DISEASE_CLASSES, pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    labels, probs, ids, pids = evaluate(model, loader, device)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None,
                         labels=list(range(NUM_DISEASE_CLASSES)), zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_DISEASE_CLASSES)))
    p_metrics = patient_level_metrics(pids, labels, preds, probs=probs)

    print("\n" + "=" * 60)
    print("疾病分类评估结果")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"AUC (OvR): {auc:.4f}")
    print("\n逐类 F1:")
    for i, name in enumerate(DISEASE_CLASSES):
        print(f"  {name:22s} ({DISEASE_CLASSES_ZH[name]}): {per_class[i]:.4f}")
    print("\n混淆矩阵 (行=真实, 列=预测):")
    print("            " + " ".join(f"{n:>10}" for n in DISEASE_CLASSES))
    for i, row in enumerate(cm):
        print(f"  {DISEASE_CLASSES[i]:9s} " + " ".join(f"{v:10d}" for v in row))

    print("\n" + "-" * 60)
    print(f"患者级评估（多数投票，共 {p_metrics['n_patients']} 名患者）")
    print("-" * 60)
    print(f"患者级 Accuracy:      {p_metrics['accuracy']:.4f}")
    print(f"患者级 Macro F1:      {p_metrics['macro_f1']:.4f}")
    print(f"患者级 AUC:           {p_metrics['auc']:.4f}")
    print(f"全对患者占比 (All-correct): "
          f"{p_metrics['all_correct_count']}/{p_metrics['n_patients']} = "
          f"{p_metrics['all_correct_accuracy']:.4f}")
    print("\n患者级逐类 F1:")
    for i, name in enumerate(DISEASE_CLASSES):
        print(f"  {name:22s} ({DISEASE_CLASSES_ZH[name]}): "
              f"{p_metrics['per_class_f1'][i]:.4f}")
    print("\n患者级混淆矩阵 (行=真实, 列=预测):")
    print("            " + " ".join(f"{n:>10}" for n in DISEASE_CLASSES))
    for i, row in enumerate(p_metrics["confusion_matrix"]):
        print(f"  {DISEASE_CLASSES[i]:9s} " + " ".join(f"{v:10d}" for v in row))

    # 逐图预测（可选保存）
    per_image = [{
        "image_id": ids[i], "true": int(labels[i]), "pred": int(preds[i]),
        "true_name": DISEASE_CLASSES[int(labels[i])],
        "pred_name": DISEASE_CLASSES[int(preds[i])],
        "confidence": float(probs[i][preds[i]]),
    } for i in range(len(ids))]

    report = {
        "checkpoint": str(ckpt_path),
        "backbone": backbone,
        "image_size": image_size,
        "num_samples": len(test_samples),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "auc": auc,
        "per_class_f1": {DISEASE_CLASSES[i]: float(per_class[i])
                         for i in range(NUM_DISEASE_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "patient_level": {
            "n_patients": p_metrics["n_patients"],
            "accuracy": p_metrics["accuracy"],
            "macro_f1": p_metrics["macro_f1"],
            "auc": p_metrics["auc"],
            "all_correct_accuracy": p_metrics["all_correct_accuracy"],
            "all_correct_count": p_metrics["all_correct_count"],
            "per_class_f1": {DISEASE_CLASSES[i]:
                             float(p_metrics["per_class_f1"][i])
                             for i in range(NUM_DISEASE_CLASSES)},
            "confusion_matrix": p_metrics["confusion_matrix"].tolist(),
        },
    }
    out = OUTPUT_DIR / "evaluation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"metrics": report, "per_image": per_image},
                  f, indent=2, ensure_ascii=False)
    print(f"\n评估报告已保存: {out}")


if __name__ == "__main__":
    main()
