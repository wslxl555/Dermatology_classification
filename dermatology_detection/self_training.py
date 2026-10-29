"""
自训练 / 伪标签迭代：用 unlabeled 数据反向修正「疾病三分类」模型。

思路（经典 self-training）：
  1. 在 labeled/ 上训练初始分类模型（按患者分层切 train/val）。
  2. 对 unlabeled/（自训练池）逐张预测，选出高置信样本作为「伪标签」。
  3. 把伪标签样本并入训练集，微调模型。
  4. 重复 2-3 若干轮，每轮在留出的测试集（unlabeled 的 test_size 比例）上评估。

关键设计：
  - unlabeled 按疾病分目录，目录名本身就是可靠真值。为演示真正的「自训练」，
    伪标签取自**模型预测**（而非目录名）；但额外用目录名作为「静默真值」统计
    伪标签正确率，以便判断自训练灌入的是正确还是错误样本。
  - 训练集每轮只增不减；模型跨轮「热启动」（保留上一轮权重继续微调）。
  - 类别不平衡：可选按类限制伪标签数量，避免模型偏置被放大。

用法:
    python self_training.py                                # 默认 CPU 安全配置
    python self_training.py --image_size 256 --iterations 5
    python self_training.py --tiny                          # 快速自检
    python self_training.py --offline                       # 仅用缓存权重
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import DISEASE_CLASSES, DISEASE_CLASSES_ZH, OUTPUT_DIR, CHECKPOINT_DIR
from data.prepare_labeled_data import scan_labeled_dataset, DISEASE_TO_IDX
from data.disease_cls_dataset import DiseaseClassificationDataset
from data.transforms import get_cls_train_transforms, get_cls_val_transforms
from models.classification_model import build_classification_model


def parse_args():
    p = argparse.ArgumentParser(description="Self-training / pseudo-labeling for disease classification")
    p.add_argument("--labeled_root", type=str, default=None,
                   help="labeled/ 目录（默认 ../labeled）")
    p.add_argument("--unlabeled_root", type=str, default=None,
                   help="unlabeled/ 目录（默认 ../unlabeled）")
    p.add_argument("--backbone", type=str, default="convnext_tiny",
                   choices=["convnext_tiny", "convnext_small", "resnet50", "swin_t"])
    p.add_argument("--image_size", type=int, default=224, help="正方形输入尺寸")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--initial_epochs", type=int, default=10, help="初始训练 epoch 数")
    p.add_argument("--epochs_per_iter", type=int, default=5, help="每轮微调 epoch 数")
    p.add_argument("--iterations", type=int, default=3, help="自训练轮数")
    p.add_argument("--confidence_threshold", type=float, default=0.9,
                   help="伪标签置信度阈值（最大 softmax 概率，threshold 模式）")
    p.add_argument("--max_pseudo_per_class", type=int, default=None,
                   help="每类每轮最多新增的伪标签数（None=不限）")
    p.add_argument("--per_class_topk", type=int, default=None,
                   help="按类均衡选择：每类取 top-K 最自信预测（优先于阈值法，"
                        "专治弱类如荨麻疹选不上伪标签的问题）")
    p.add_argument("--min_conf", type=float, default=0.5,
                   help="per_class_topk 模式下的最低置信度下限")
    p.add_argument("--resume_from", type=str, default=None,
                   help="从已有纯分类 checkpoint 热启动（如 disease_cls_best.pth），"
                        "跳过初始训练")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_size", type=float, default=0.15, help="labeled 内切出的验证比例")
    p.add_argument("--test_size", type=float, default=0.2, help="unlabeled 留出的测试比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augmentation", type=str, default="medical",
                   choices=["medical", "standard", "none"])
    p.add_argument("--offline", action="store_true", help="仅用缓存权重")
    p.add_argument("--tiny", action="store_true", help="快速自检（小规模）")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 数据工具
# ---------------------------------------------------------------------------
def patient_split(samples, test_size, seed):
    """按 (disease, patient_id) 分层切分，保证同一患者不同时出现在两侧。"""
    groups = defaultdict(list)
    for s in samples:
        groups[(s["disease"], s["patient_id"])].append(s)
    keys = list(groups.keys())
    y = [DISEASE_TO_IDX[k[0]] for k in keys]

    counts = Counter(k[0] for k in keys)
    stratify = y if min(counts.values()) >= 2 else None
    if test_size <= 0:
        return samples, []
    a_keys, b_keys = train_test_split(
        keys, test_size=test_size, stratify=stratify, random_state=seed)
    a_set, b_set = set(a_keys), set(b_keys)
    a = [s for s in samples if (s["disease"], s["patient_id"]) in a_set]
    b = [s for s in samples if (s["disease"], s["patient_id"]) in b_set]
    return a, b


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["disease_label"] for b in batch], dtype=torch.long)
    image_ids = [b["image_id"] for b in batch]
    return {"image": images, "disease_label": labels, "image_id": image_ids}


def class_weights_from(samples):
    counts = Counter(s["disease"] for s in samples)
    w = torch.zeros(len(DISEASE_CLASSES))
    for i, name in enumerate(DISEASE_CLASSES):
        w[i] = 1.0 / max(counts.get(name, 0), 1)
    return w / w.sum() * len(DISEASE_CLASSES)


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, n = 0.0, 0
    labels_all, probs_all = [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        y = batch["disease_label"].to(device)
        if train:
            optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
        probs_all.append(torch.softmax(logits, 1).detach().cpu().numpy())
        labels_all.append(y.detach().cpu().numpy())
    return total / max(n, 1), np.concatenate(labels_all), np.concatenate(probs_all)


def compute_metrics(labels, probs):
    preds = probs.argmax(1)
    nc = len(DISEASE_CLASSES)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None,
                         labels=list(range(nc)), zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(labels, preds, labels=list(range(nc)))
    return {"acc": acc, "macro_f1": macro_f1, "per_class_f1": per_class,
            "auc": auc, "cm": cm}


def _fmt_metrics(tag, m):
    s = f"  {tag:5s} | Acc {m['acc']:.4f} | MacroF1 {m['macro_f1']:.4f} | AUC {m['auc']:.4f}"
    s += " | " + " ".join(f"{DISEASE_CLASSES[i]}={m['per_class_f1'][i]:.3f}"
                          for i in range(len(DISEASE_CLASSES)))
    return s


@torch.no_grad()
def predict_pool(model, samples, args, device):
    """对样本列表预测，返回 [(sample, prob, pred_class)] 按置信度降序。"""
    tf = get_cls_val_transforms((args.image_size, args.image_size))
    ds = DiseaseClassificationDataset(samples, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    model.eval()
    results = []
    for batch in loader:
        imgs = batch["image"].to(device)
        probs = torch.softmax(model(imgs), 1).cpu().numpy()
        for i in range(probs.shape[0]):
            results.append((batch["image_id"][i], probs[i]))
    return results  # [(image_id, prob_vec)]


def train_round(model, train_samples, val_samples, args, device, epochs):
    """微调 `epochs` 个 epoch，用 val 早停，返回 (model, best_val_macro_f1)。

    每轮重建 DataLoader（样本数会变），但模型权重跨轮保留（热启动）。
    """
    train_tf = get_cls_train_transforms((args.image_size, args.image_size), level=args.augmentation)
    val_tf = get_cls_val_transforms((args.image_size, args.image_size))

    train_ds = DiseaseClassificationDataset(train_samples, transform=train_tf)
    val_ds = DiseaseClassificationDataset(val_samples, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # 差异化学习率：backbone 小 LR，分类头 10x
    head = [p for n, p in model.named_parameters() if n.startswith("head.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW([
        {"params": body, "lr": args.lr},
        {"params": head, "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    cw = class_weights_from(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_f1, best_state = 0.0, None
    for ep in range(epochs):
        t0 = time.time()
        run_epoch(model, train_loader, device, criterion, optimizer)
        scheduler.step()
        _, vl, vp = run_epoch(model, val_loader, device, criterion)
        vm = compute_metrics(vl, vp)
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"      ep {ep} | {time.time()-t0:5.1f}s | {_fmt_metrics('val', vm)}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def main():
    args = parse_args()
    if args.tiny:
        args.image_size = 224
        args.initial_epochs = 2
        args.epochs_per_iter = 1
        args.iterations = 1
        args.batch_size = 4
        print("*** TINY MODE — quick sanity check ***")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    labeled_root = Path(args.labeled_root) if args.labeled_root else _project_root.parent / "labeled"
    unlabeled_root = Path(args.unlabeled_root) if args.unlabeled_root else _project_root.parent / "unlabeled"

    # 1) 扫描
    print("Scanning labeled ...")
    labeled = scan_labeled_dataset(labeled_root)
    print("Scanning unlabeled ...")
    unlabeled = scan_labeled_dataset(unlabeled_root)
    print(f"labeled: {len(labeled)} | unlabeled: {len(unlabeled)}")

    # 2) 划分
    labeled_train, labeled_val = patient_split(labeled, test_size=args.val_size, seed=args.seed)
    unlabeled_pool, test = patient_split(unlabeled, test_size=args.test_size, seed=args.seed)
    print(f"labeled train {len(labeled_train)} | val {len(labeled_val)} | "
          f"unlabeled pool {len(unlabeled_pool)} | test {len(test)}")
    print(f"  test dist: {dict(Counter(s['disease'] for s in test))}")

    # 3) 初始模型
    model = build_classification_model(
        backbone_name=args.backbone, pretrained=True, offline=args.offline).to(device)
    print(f"Model {args.backbone} | params "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M | device {device}")

    # 4) 初始训练（或用已有 checkpoint 热启动）
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"热启动自 {args.resume_from}（best_val_f1={ck.get('best_val_f1', '?')}），"
              f"跳过初始训练。")
    else:
        print(f"\n>>> 初始训练 on labeled（{args.initial_epochs} epochs）")
        model, _ = train_round(model, labeled_train, labeled_val, args, device, args.initial_epochs)

    # 5) 初始测试
    _, te_l, te_p = run_epoch(model, DataLoader(
        DiseaseClassificationDataset(test, transform=get_cls_val_transforms((args.image_size, args.image_size))),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn),
        device, nn.CrossEntropyLoss())
    base_m = compute_metrics(te_l, te_p)
    print(f"\n[基线] 初始模型测试集: {_fmt_metrics('test', base_m)}")

    # 6) 自训练迭代
    history = [{"round": "baseline", "test": {k: (v.tolist() if hasattr(v, 'tolist') else v)
               for k, v in base_m.items()}}]
    pool = list(unlabeled_pool)
    cur_train = list(labeled_train)

    for it in range(args.iterations):
        print(f"\n{'='*60}\n>>> 自训练轮 {it+1}/{args.iterations}")
        if not pool:
            print("  pool 已空，提前结束。")
            break

        # 预测整个剩余 pool
        preds = predict_pool(model, pool, args, device)
        id2sample = {Path(s["image_path"]).stem: s for s in pool}

        selected = []
        per_class = Counter()
        pseudo_correct = 0

        if args.per_class_topk:
            # 按类均衡选择：每类取 top-K 最自信预测（argmax==该类），
            # 保证弱类（荨麻疹）也能拿到伪标签，避免被湿疹淹没。
            by_class = {c: [] for c in range(len(DISEASE_CLASSES))}
            for image_id, prob in preds:
                c = int(prob.argmax())
                conf = float(prob.max())
                if conf >= args.min_conf:
                    by_class[c].append((image_id, conf))
            for c in range(len(DISEASE_CLASSES)):
                by_class[c].sort(key=lambda x: x[1], reverse=True)
                for image_id, conf in by_class[c][:args.per_class_topk]:
                    s = id2sample[image_id]
                    selected.append({**s, "disease": DISEASE_CLASSES[c],
                                     "pseudo_conf": conf, "data_quality": "pseudo"})
                    per_class[c] += 1
                    if DISEASE_CLASSES[c] == s["disease"]:
                        pseudo_correct += 1
        else:
            # 全局置信度阈值法（原逻辑）
            for image_id, prob in sorted(preds, key=lambda x: x[1].max(), reverse=True):
                conf = float(prob.max())
                pred_cls = int(prob.argmax())
                if conf < args.confidence_threshold:
                    continue
                if args.max_pseudo_per_class and per_class[pred_cls] >= args.max_pseudo_per_class:
                    continue
                s = id2sample[image_id]
                selected.append({**s, "disease": DISEASE_CLASSES[pred_cls],
                                 "pseudo_conf": conf, "data_quality": "pseudo"})
                per_class[pred_cls] += 1
                if DISEASE_CLASSES[pred_cls] == s["disease"]:
                    pseudo_correct += 1

        mode = (f"per_class_topk={args.per_class_topk}, min_conf={args.min_conf}"
                if args.per_class_topk else f"阈值≥{args.confidence_threshold}")
        print(f"  选中伪标签 {len(selected)} 张（{mode}）")
        print(f"  伪标签分布: {dict(per_class)}")
        if selected:
            print(f"  伪标签正确率（对照目录真值）: {pseudo_correct/len(selected):.3f}")
        else:
            print("  无满足阈值的样本，提前结束。")
            break

        # 加入训练集并微调
        cur_train = cur_train + selected
        # 从 pool 中移除已选中的（避免重复）
        selected_ids = {Path(s["image_path"]).stem for s in selected}
        pool = [s for s in pool if Path(s["image_path"]).stem not in selected_ids]

        print(f"  当前训练样本 {len(cur_train)}（labeled {len(labeled_train)} + 伪标签 "
              f"{len(cur_train)-len(labeled_train)}）")
        model, best_f1 = train_round(model, cur_train, labeled_val, args, device, args.epochs_per_iter)

        # 评估测试集
        _, te_l, te_p = run_epoch(model, DataLoader(
            DiseaseClassificationDataset(test, transform=get_cls_val_transforms((args.image_size, args.image_size))),
            batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn),
            device, nn.CrossEntropyLoss())
        te_m = compute_metrics(te_l, te_p)
        print(f"  [轮 {it+1}] 测试集: {_fmt_metrics('test', te_m)}")

        history.append({"round": f"iter_{it+1}", "n_pseudo": len(selected),
                        "pseudo_dist": dict(per_class),
                        "pseudo_acc": (pseudo_correct/len(selected) if selected else None),
                        "train_n": len(cur_train),
                        "test": {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                 for k, v in te_m.items()}})

    # 7) 保存
    ckpt_path = CHECKPOINT_DIR / "selftrain_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"backbone": args.backbone, "image_size": args.image_size,
                   "num_classes": len(DISEASE_CLASSES), "class_names": DISEASE_CLASSES},
        "history": history,
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    hist_path = OUTPUT_DIR / "selftrain_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str, ensure_ascii=False)
    print(f"History saved: {hist_path}")

    # 汇总对比
    print("\n" + "=" * 60)
    print("自训练前后测试集对比")
    print("=" * 60)
    for h in history:
        t = h["test"]
        print(f"  {h['round']:10s} | Acc {t['acc']:.4f} | MacroF1 {t['macro_f1']:.4f} "
              f"| AUC {t['auc']:.4f}")
    print("=" * 60)
    print("完成。")


if __name__ == "__main__":
    main()
