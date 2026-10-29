"""
主训练入口。

用法:
    python train.py                          # 完整训练
    python train.py --tiny                    # 快速验证
    python train.py --medium                  # CPU 本地训练(中等配置)
    python train.py --exp 1                   # 运行预设实验 (1-6)
    python train.py --exp 1 --data_tier all   # 实验 + 数据层级
    python train.py --epochs 30 --lr 5e-5    # 自定义超参数
    python train.py --resume best.pth         # 从 checkpoint 恢复训练
    python train.py --data_root D:/images     # 指定数据目录
"""

import argparse
import math
import pickle
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    FULL_CONFIG, TINY_CONFIG, MEDIUM_CPU_CONFIG,
    DATA_ROOT, OUTPUT_DIR, CHECKPOINT_DIR,
    NUM_POSITION_CLASSES,
)
from data.prepare_data import scan_dataset, create_splits
from data.dataset import EczemaDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.multi_task_model import build_model
from training.trainer import Trainer


def _make_serializable(obj):
    """递归地将 NaN 转为 None,将 tensor 转为 float,使其可 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    elif hasattr(obj, "item"):
        val = obj.item()
        return None if (isinstance(val, float) and math.isnan(val)) else val
    return obj


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train skin disease classifier")
    parser.add_argument("--tiny", action="store_true",
                        help="Quick sanity check with reduced config")
    parser.add_argument("--medium", action="store_true",
                        help="CPU-friendly medium config for local training")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override DATA_ROOT path (e.g., for local training)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["convnext_tiny", "convnext_small", "resnet50", "swin_t"])
    parser.add_argument("--no_detection", action="store_true",
                        help="Disable lesion detection branch")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (e.g., best.pth)")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Image size (single int for square)")
    parser.add_argument("--augmentation", type=str, default=None,
                        choices=["medical", "standard", "none"])
    parser.add_argument("--offline", action="store_true",
                        help="Skip network access; use cached weights only")
    parser.add_argument("--exp", type=int, default=None, choices=[1, 2, 3, 4, 5, 6],
                        help="Run a preset experiment (1-6)")
    parser.add_argument("--data_tier", type=str, default=None,
                        choices=["入选", "all"],
                        help="Data quality tier: 入选 (high-quality only) or all")
    return parser.parse_args()


def main():
    """训练主入口:准备数据、构建模型并执行训练。"""
    args = parse_args()

    # 配置
    if args.exp:
        from config import EXP_CONFIG_MAP
        config = EXP_CONFIG_MAP[args.exp].copy()
        print(f"*** Experiment {args.exp}: {config.get('exp_name', 'N/A')} ***")
    elif args.tiny:
        config = TINY_CONFIG
        print("*** TINY MODE — quick sanity check ***")
    elif args.medium:
        config = MEDIUM_CPU_CONFIG
        print("*** MEDIUM CPU MODE — local training ***")
    else:
        config = FULL_CONFIG.copy()

    # data_tier 覆盖
    if args.data_tier:
        config["data_tier"] = args.data_tier

    # 根据命令行参数覆盖配置（优先级高于预设实验）
    if args.epochs:
        config["max_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.image_size:
        config["image_size"] = (args.image_size, args.image_size)
    if args.augmentation:
        config["augmentation"] = args.augmentation
    if args.no_detection:
        config["detection_enabled"] = False

    # 覆盖 DATA_ROOT(用于本地训练)
    import config as cfg
    if args.data_root:
        new_root = Path(args.data_root).resolve()
        print(f"Overriding DATA_ROOT: {cfg.DATA_ROOT} -> {new_root}")
        cfg.DATA_ROOT = new_root
        # 重建 DISEASE_DIR_MAP (3 类)
        cfg.DISEASE_DIR_MAP = {
            "eczema": new_root / "sz",
            "bullous_pemphigoid": new_root / "tpc",
            "urticaria": new_root / "xmz",
        }

    # 加载或创建数据集划分
    splits_path = OUTPUT_DIR / "splits.pkl"
    need_rescan = not splits_path.exists()
    if not need_rescan:
        print(f"Loading existing splits from {splits_path}")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        # 校验:检查划分中的路径是否真实存在。
        # 来自其他操作系统(如 Windows → Linux)的旧划分会带有
        # 过期的绝对路径——此时强制重新扫描。
        from data.dataset import _resolve_path
        first_image = splits["train"][0]["image_path"]
        resolved = _resolve_path(first_image)
        if not Path(resolved).exists():
            print(f"  [WARN] Cached splits contain stale paths "
                  f"(e.g. '{first_image}' → '{resolved}' not found).")
            print(f"  Re-scanning dataset on this machine …")
            need_rescan = True
    if need_rescan:
        data_tier = config.get("data_tier", "入选")
        print(f"Scanning dataset and creating new splits (data_tier={data_tier})...")
        samples = scan_dataset(data_tier=data_tier)
        splits = create_splits(
            samples,
            val_size=config["val_split"],
            test_size=0.1,
            seed=config["seed"],
            use_kfold=False,
            min_val_per_class=2,
            tiny_class_threshold=10,
            tiny_class_val_ratio=0.30,
        )
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)

    train_samples = splits["train"]
    val_samples = splits["val"]

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    # 数据变换
    train_transform = get_train_transforms(
        image_size=config["image_size"],
        level=config["augmentation"],
    )
    val_transform = get_val_transforms(
        image_size=config["image_size"],
    )

    # 数据集
    train_dataset = EczemaDataset(train_samples, transform=train_transform)
    val_dataset = EczemaDataset(val_samples, transform=val_transform)

    # 计算训练集类别分布，创建加权采样器以解决类别不平衡 (3 类)
    disease_to_idx = {"eczema": 0, "bullous_pemphigoid": 1, "urticaria": 2}
    train_labels = [disease_to_idx[s["disease"]] for s in train_samples]
    class_counts = Counter(train_labels)
    print(f"Train class distribution (label: count): "
          f"{ {d: class_counts.get(i, 0) for i, d in enumerate(disease_to_idx)} }")

    # 逆频率权重：每个样本权重 = 1 / 该类别样本数
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # 数据加载器
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,  # 使用加权采样器替代 shuffle（二者互斥）
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )

    # 模型
    backbone = args.backbone or config.get("backbone", "convnext_tiny")
    model = build_model(
        backbone_name=backbone,
        pretrained=config["pretrained"],
        detection_enabled=config.get("detection_enabled", False),
        offline=args.offline,
    )

    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # 如请求则恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 打印实验配置摘要
    exp_name = config.get("exp_name", "custom")
    print(f"\n{'='*60}")
    print(f"EXPERIMENT CONFIG: {exp_name}")
    print(f"{'='*60}")
    print(f"  Backbone:      {args.backbone or config.get('backbone', 'convnext_tiny')}")
    print(f"  Image size:    {config['image_size']}")
    print(f"  Batch size:    {config['batch_size']}")
    print(f"  Max epochs:    {config['max_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Augmentation:  {config['augmentation']}")
    print(f"  Detection:     {config.get('detection_enabled', False)}")
    print(f"  Data tier:     {config.get('data_tier', '入选')}")
    print(f"  Position head: True (4 classes: L/S/J/B)")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples:   {len(val_samples)}")
    print(f"{'='*60}\n")

    # 训练
    history = trainer.fit()

    # 保存训练历史
    import json
    history_path = OUTPUT_DIR / "training_history.json"
    serializable_history = _make_serializable(history)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2, default=str, ensure_ascii=False)
    print(f"History saved to {history_path}")

    # 最终评估
    print("\n" + "=" * 60)
    print("FINAL VALIDATION")
    print(trainer.val_metrics.get_classification_report())


def _collate_fn(batch: list[dict]) -> dict:
    """针对变长 bbox 的自定义 collate 函数。"""
    images = torch.stack([item["image"] for item in batch])
    disease_labels = torch.tensor([item["disease_label"] for item in batch])
    position_labels = torch.tensor([item["position_label"] for item in batch])
    lesion_multi_hot = torch.stack([item["lesion_multi_hot"] for item in batch])
    bboxes = [item["bboxes"] for item in batch]
    lesion_labels = [item["lesion_labels"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    return {
        "image": images,
        "disease_label": disease_labels,
        "position_label": position_labels,
        "lesion_multi_hot": lesion_multi_hot,
        "bboxes": bboxes,
        "lesion_labels": lesion_labels,
        "image_id": image_ids,
    }


if __name__ == "__main__":
    # 修复 Python 3.14+ 在 Windows 上的多进程问题
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
