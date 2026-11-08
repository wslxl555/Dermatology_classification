"""
训练病灶「形态」分类器（crop 级）。

从 labeled/ 的 LabelMe 标注中裁剪病灶，训练一个 CNN 判断每个病灶的形态
（丘疹/斑块/水疱/大疱/鳞屑…）。它是「病灶 crop 分类 → 疾病」两段式里的第一段。

复用上层代码：build_model（backbone + 分类头）、run_epoch / compute_metrics（train.py）。

用法:
    python train_lesion_classifier.py --backbone swin_t --image_size 224 --epochs 20 --offline
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEFAULT_CONFIG, OUTPUT_DIR, CHECKPOINT_DIR
from transforms import get_train_transforms, get_val_transforms
from model import build_model, BACKBONE_NAMES
from train import run_epoch, compute_metrics
from lesion_crops import build_crop_dataset, MORPHOLOGY_CLASSES

NUM_MORPH = len(MORPHOLOGY_CLASSES)


def parse_args():
    p = argparse.ArgumentParser(description="训练病灶形态分类器")
    p.add_argument("--labeled_root", type=str, default=None)
    p.add_argument("--backbone", type=str, default=DEFAULT_CONFIG["backbone"],
                   choices=BACKBONE_NAMES)
    p.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--backbone_lr", type=float, default=DEFAULT_CONFIG["backbone_lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    p.add_argument("--label_smoothing", type=float,
                   default=DEFAULT_CONFIG["label_smoothing"])
    p.add_argument("--min_size", type=int, default=8,
                   help="过滤边长小于该值的病灶")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--offline", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def crop_collate(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {"image": images, "label": labels}


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, crops, labels, transform=None):
        self.crops = crops
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        img = self.crops[i]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, self.labels[i]


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    crops, labels, metas = build_crop_dataset(args.labeled_root, args.min_size)
    print(f"共裁剪 {len(crops)} 个病灶 crop，{NUM_MORPH} 类形态")
    dist = Counter(MORPHOLOGY_CLASSES[l] for l in labels)
    for name, c in dist.most_common():
        print(f"  {name:16s}: {c}")

    if len(crops) < 16:
        raise RuntimeError("病灶 crop 太少，无法训练")

    # 随机划分（按 crop；同一患者/图片的 crop 可能跨 train/val，形态分类仅作中间模型）
    n = len(crops)
    idx = np.random.permutation(n)
    val_n = max(int(n * 0.15), 1)
    val_idx, tr_idx = idx[:val_n], idx[val_n:]
    train_crops = [crops[i] for i in tr_idx]
    train_labels = [labels[i] for i in tr_idx]
    val_crops = [crops[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_tf = get_train_transforms(args.image_size, level="medical")
    val_tf = get_val_transforms(args.image_size)
    train_loader = DataLoader(CropDataset(train_crops, train_labels, train_tf),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=crop_collate)
    val_loader = DataLoader(CropDataset(val_crops, val_labels, val_tf),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=crop_collate)

    model = build_model(
        backbone_name=args.backbone, num_classes=NUM_MORPH,
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
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    print(f"\n模型: {args.backbone} | 形态类别 {NUM_MORPH} | 设备 {device}")
    best_f1 = 0.0
    for epoch in range(args.epochs):
        run_epoch(model, train_loader, device, criterion, optimizer)
        scheduler.step()
        _, va_labels, va_probs = run_epoch(model, val_loader, device, criterion)
        va_m = compute_metrics(va_labels, va_probs)
        if va_m["macro_f1"] > best_f1:
            best_f1 = va_m["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "morphology_classes": MORPHOLOGY_CLASSES,
                "backbone": args.backbone,
                "image_size": args.image_size,
            }, CHECKPOINT_DIR / "lesion_morph.pth")
        print(f"Epoch {epoch:3d} | Val Acc {va_m['acc']:.4f} | "
              f"Macro-F1 {va_m['macro_f1']:.4f}")

    print(f"\n最佳 val macro-F1: {best_f1:.4f}")
    print(f"形态分类器已保存: {CHECKPOINT_DIR / 'lesion_morph.pth'}")


if __name__ == "__main__":
    main()
