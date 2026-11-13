"""
训练病灶检测器（可自由切换检测头）。

检测头可选：faster_rcnn / retinanet / fcos（torchvision 内置，接口统一）。

用法:
    python detector/train_detector.py --head fcos
    python detector/train_detector.py --head faster_rcnn --image_size 1024 --epochs 30
    python detector/train_detector.py --head retinanet --offline
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from sklearn.model_selection import train_test_split

from config import LABELED_ROOT, CHECKPOINT_DIR, OUTPUT_DIR
from detector.detection_model import (
    DETECTION_CLASSES, DETECTION_HEAD_NAMES, NUM_MODEL_CLASSES,
    build_detector, scan_detection_samples, LesionDetectionDataset, collate_fn,
)


def parse_args():
    p = argparse.ArgumentParser(description="训练病灶检测器")
    p.add_argument("--labeled_root", type=str, default=str(LABELED_ROOT))
    p.add_argument("--head", type=str, default="fcos", choices=DETECTION_HEAD_NAMES)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline", action="store_true", help="只用缓存权重（resnet50 需先缓存）")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def patient_split(samples, val_size, seed):
    groups = defaultdict(list)
    for s in samples:
        groups[s["patient_id"]].append(s)
    pids = list(groups.keys())
    train_pids, val_pids = train_test_split(pids, test_size=val_size,
                                            random_state=seed)
    val_set = set(val_pids)
    train = [s for s in samples if s["patient_id"] not in val_set]
    val = [s for s in samples if s["patient_id"] in val_set]
    return train, val


@torch.no_grad()
def eval_detection(model, loader, device, score_thresh=0.05, iou_thresh=0.5):
    """简化检测评估：GT 框被正确类别且 IoU>阈值的预测命中的召回/精确率。"""
    model.eval()
    n_gt = n_pred = n_matched = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        for target, pred in zip(targets, preds):
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            p_boxes = pred["boxes"]
            p_labels = pred["labels"]
            p_scores = pred["scores"]
            keep = p_scores >= score_thresh
            p_boxes, p_labels = p_boxes[keep], p_labels[keep]
            n_gt += len(gt_boxes)
            n_pred += len(p_boxes)
            if len(gt_boxes) == 0 or len(p_boxes) == 0:
                continue
            iou = box_iou(gt_boxes, p_boxes)  # [N_gt, N_pred]
            used = set()
            for gi in range(len(gt_boxes)):
                best_j, best_iou = -1, iou_thresh
                for pj in range(len(p_boxes)):
                    if pj in used or p_labels[pj] != gt_labels[gi]:
                        continue
                    if iou[gi, pj] > best_iou:
                        best_j, best_iou = pj, iou[gi, pj].item()
                if best_j >= 0:
                    used.add(best_j)
                    n_matched += 1
    recall = n_matched / max(n_gt, 1)
    precision = n_matched / max(n_pred, 1)
    return recall, precision, n_gt, n_pred


def save_ckpt(model, head, image_size, epoch, ckpt_path):
    """保存检测器 checkpoint（字段与 load_detector 完全对齐）。"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "detection_head": head,
        "detection_classes": DETECTION_CLASSES,
        "image_size": image_size,
    }, ckpt_path)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    samples = scan_detection_samples(args.labeled_root)
    print(f"扫描到 {len(samples)} 张带标注图片")
    train, val = patient_split(samples, args.val_size, args.seed)
    print(f"train={len(train)} 张, val={len(val)} 张")

    train_ds = LesionDetectionDataset(train, (args.image_size, args.image_size), flip=True)
    val_ds = LesionDetectionDataset(val, (args.image_size, args.image_size), flip=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    model = build_detector(args.head, NUM_MODEL_CLASSES, offline=args.offline).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n模型: {args.head} @ {args.image_size}px | 检测类别 {len(DETECTION_CLASSES)} | {device}")

    best_recall = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batch = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(v for v in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
            n_batch += 1
        scheduler.step()

        recall, precision, n_gt, n_pred = eval_detection(model, val_loader, device)
        avg_loss = total_loss / max(n_batch, 1)
        print(f"Epoch {epoch:3d} | {time.time()-t0:5.1f}s | loss {avg_loss:.4f} | "
              f"val recall@{0.5:.0f}={recall:.4f} precision={precision:.4f} "
              f"(GT {n_gt}, pred {n_pred})")

        if recall > best_recall:
            best_recall = recall
            save_ckpt(model, args.head, args.image_size, epoch,
                      CHECKPOINT_DIR / "lesion_detector.pth")
            print(f"  >> 已保存最佳检测器 (val recall={recall:.4f})")

    # 兜底：若全程 recall 都为 0（如离线随机初始化、模型一个框都没检出），
    # 上面的分支一次都不触发，会导致没有 checkpoint、Step 6 直接 FileNotFoundError。
    ckpt_path = CHECKPOINT_DIR / "lesion_detector.pth"
    if not ckpt_path.exists():
        save_ckpt(model, args.head, args.image_size, args.epochs - 1, ckpt_path)
        print(f"  [警告] val recall 全程为 0，已保存最后一轮模型（仅保证可运行，质量可能较差）")

    print(f"\n最佳 val recall@0.5: {best_recall:.4f}")
    print(f"检测器已保存: {ckpt_path}")


if __name__ == "__main__":
    main()
