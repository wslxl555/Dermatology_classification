"""
使用 labeled/ 文件夹中的标注数据训练多任务模型（疾病分类 + 皮损检测）。

用法:
    python train_labeled.py                          # 完整训练
    python train_labeled.py --tiny                    # 快速验证
    python train_labeled.py --medium                  # CPU 本地训练
    python train_labeled.py --epochs 30 --lr 5e-5    # 自定义超参数
    python train_labeled.py --backbone resnet50       # 切换主干网络
    python train_labeled.py --no_detection            # 仅疾病分类（不训练检测头）
    python train_labeled.py --image_size 256          # 自定义图像尺寸
"""

import argparse
import math
import pickle
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# 动态添加项目根目录到 path
import sys
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    FULL_CONFIG, TINY_CONFIG, MEDIUM_CPU_CONFIG,
    OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR,
    DISEASE_CLASSES, DISEASE_CLASSES_ZH, DETECTION_HEAD_NAMES,
)
from models.backbone import BACKBONE_NAMES
from data.prepare_labeled_data import (
    scan_labeled_dataset,
    create_labeled_splits,
    DISEASE_TO_IDX,
)
from data.dataset import EczemaDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.multi_task_model import build_model
from training.trainer import Trainer


# =============================================================================
# 基于 labeled 数据的默认配置
# =============================================================================
LABELED_TRAIN_CONFIG = {
    **FULL_CONFIG,
    # 数据
    "image_size": (512, 512),
    "batch_size": 4,           # 数据量较小，减小 batch
    "num_workers": 0,          # Windows 兼容
    "augmentation": "medical",

    # 训练
    "max_epochs": 50,
    "early_stopping_patience": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "val_split": 0.15,
    "seed": 42,

    # 模型
    "backbone": "convnext_tiny",
    "pretrained": True,
    "detection_enabled": True,
    "fpn_channels": 256,

    # 损失
    "loss_disease_weight": 1.0,
    "loss_detection_weight": 1.0,
    "loss_position_weight": 0.0,  # labeled 数据无位置信息
    "num_position_classes": 4,
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,

    # 混合精度
    "use_amp": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# 快速验证配置
LABELED_TINY_CONFIG = {
    **LABELED_TRAIN_CONFIG,
    "image_size": (224, 224),
    "batch_size": 4,
    "num_workers": 0,
    "max_epochs": 3,
    "detection_enabled": False,
    "use_amp": False,
}

# CPU 训练配置
LABELED_MEDIUM_CONFIG = {
    **LABELED_TRAIN_CONFIG,
    "device": "cpu",
    "image_size": (224, 224),
    "batch_size": 4,
    "num_workers": 0,
    "max_epochs": 30,
    "early_stopping_patience": 10,
    "detection_enabled": False,
    "use_amp": False,
}

# 安全 CPU 训练配置（本机 8GB 内存、无 GPU）。
# 启用检测分支 + ConvNeXt-Tiny 在 256px / batch2 下约 12–15 分钟/epoch；
# 更高的 image_size 或 batch_size 会触发系统 OOM（训练进程被杀死）。
# 此配置在无 GPU 时作为默认值，检测分支保持启用。
LABELED_CPU_CONFIG = {
    **LABELED_TRAIN_CONFIG,
    "device": "cpu",
    "image_size": (256, 256),
    "batch_size": 2,
    "num_workers": 0,
    "max_epochs": 10,
    "early_stopping_patience": 10,
    "use_amp": False,
}


def _make_serializable(obj):
    """递归地将 NaN 转为 None,将 tensor 转为 float。"""
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
    parser = argparse.ArgumentParser(
        description="Train on labeled dermatology data with lesion detection"
    )
    parser.add_argument("--tiny", action="store_true",
                        help="Quick sanity check (3 epochs, no detection)")
    parser.add_argument("--medium", action="store_true",
                        help="CPU-friendly training")
    parser.add_argument("--labeled_root", type=str, default=None,
                        help="Path to labeled/ folder (default: ../labeled)")
    parser.add_argument("--test_size", type=float, default=0.0,
                        help="Fraction of labeled data reserved as test "
                             "(0 = use all labeled for train/val; unlabeled is the test set)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None,
                        choices=BACKBONE_NAMES)
    parser.add_argument("--detection_head", type=str, default=None,
                        choices=DETECTION_HEAD_NAMES,
                        help="Detection head (fcos / faster_rcnn / cascade_rcnn / "
                             "yolov8 / rtdetr)")
    parser.add_argument("--no_detection", action="store_true",
                        help="Disable lesion detection head")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Single int for square image size")
    parser.add_argument("--augmentation", type=str, default=None,
                        choices=["medical", "standard", "none", "tile"])
    parser.add_argument("--offline", action="store_true",
                        help="Skip network access; use cached weights only")
    parser.add_argument("--resume", type=str, default=None,
                        help="Load model weights from a checkpoint and continue "
                             "training from its next epoch (optimizer/scheduler "
                             "are re-created to save memory)")
    return parser.parse_args()


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


def main():
    args = parse_args()

    # 确定配置
    if args.tiny:
        config = LABELED_TINY_CONFIG.copy()
        print("*** TINY MODE — quick sanity check ***")
    elif args.medium:
        config = LABELED_MEDIUM_CONFIG.copy()
        print("*** MEDIUM CPU MODE — local training ***")
    else:
        # 无 GPU 时默认使用安全 CPU 配置，避免 8GB 内存 OOM；
        # 有 GPU 时使用完整训练配置（512px / batch8 / 50 epoch）。
        config = (LABELED_CPU_CONFIG if not torch.cuda.is_available()
                  else LABELED_TRAIN_CONFIG).copy()
        if not torch.cuda.is_available():
            print("*** CPU MODE — safe defaults (256px / batch2 / 10 epochs) ***")

    # 命令行参数覆盖
    if args.epochs is not None:
        config["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.image_size is not None:
        config["image_size"] = (args.image_size, args.image_size)
    if args.augmentation is not None:
        config["augmentation"] = args.augmentation
    if args.no_detection:
        config["detection_enabled"] = False

    # 确定 labeled 数据路径
    labeled_root = args.labeled_root
    if labeled_root is None:
        labeled_root = _project_root.parent / "labeled"
    labeled_root = Path(labeled_root).resolve()

    print(f"Labeled data root: {labeled_root}")

    # 加载或创建数据集划分
    # test_size=0 时把 labeled 全部用于 train/val（测试集改为 unlabeled/），
    # 使用独立的划分缓存文件，避免与旧的 15% test 划分混淆。
    test_size = args.test_size if args.test_size is not None else 0.0
    splits_name = ("labeled_splits_trainval.pkl" if test_size == 0.0
                   else "labeled_splits.pkl")
    splits_path = OUTPUT_DIR / splits_name
    need_rescan = not splits_path.exists()

    if not need_rescan:
        print(f"Loading existing splits from {splits_path}")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        # 检查划分中的图片是否存在
        first_img = splits["train"][0]["image_path"]
        if not Path(first_img).exists():
            print(f"  [WARN] Image not found: {first_img}")
            print(f"  Re-scanning labeled dataset...")
            need_rescan = True

    if need_rescan:
        print(f"Scanning labeled dataset...")
        samples = scan_labeled_dataset(labeled_root)
        print(f"Found {len(samples)} samples")

        splits = create_labeled_splits(
            samples,
            val_size=config["val_split"],
            test_size=test_size,
            seed=config["seed"],
        )

        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        print(f"Splits saved to {splits_path}")

    train_samples = splits["train"]
    val_samples = splits["val"]

    # 统计
    print(f"\nTrain: {len(train_samples)} | Val: {len(val_samples)}")
    for split_name, split_data in [("Train", train_samples), ("Val", val_samples)]:
        counts = Counter(s["disease"] for s in split_data)
        print(f"  {split_name}: {dict(counts)}")

    # 数据增强
    train_transform = get_train_transforms(
        image_size=config["image_size"],
        level=config["augmentation"],
    )
    val_transform = get_val_transforms(
        image_size=config["image_size"],
        level="tile" if config["augmentation"] == "tile" else "standard",
    )

    # 数据集
    train_dataset = EczemaDataset(train_samples, transform=train_transform)
    val_dataset = EczemaDataset(val_samples, transform=val_transform)

    # 类别加权采样器
    train_labels = [DISEASE_TO_IDX[s["disease"]] for s in train_samples]
    class_counts = Counter(train_labels)
    print(f"\nTrain class distribution: {dict(class_counts)}")

    # 真实每类样本数（用于计算 FocalLoss 的类别权重，避免用采样器口径）
    train_class_counts = [class_counts.get(i, 0)
                          for i in range(len(DISEASE_TO_IDX))]

    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # 数据加载器
    pin_memory = torch.cuda.is_available() and config.get("device", "cpu") != "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )
    # 无采样器的训练集评估 loader：用于计算真实分布下的训练混淆矩阵/Acc/F1，
    # 与验证集口径一致（训练 loader 的 WeightedRandomSampler 会把指标算在均衡分布上）。
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
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

    # 构建模型
    backbone = args.backbone or config["backbone"]
    detection_head = args.detection_head or config.get("detection_head", "fcos")
    # 将最终选择写回 config，使 checkpoint 记录 backbone / detection_head。
    config["backbone"] = backbone
    config["detection_head"] = detection_head
    model = build_model(
        backbone_name=backbone,
        pretrained=config["pretrained"],
        detection_enabled=config["detection_enabled"],
        offline=args.offline,
        detection_head=detection_head,
    )

    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        train_eval_loader=train_eval_loader,
        train_class_counts=train_class_counts,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print("NOTE: --resume loads weights only and continues from the "
              "checkpoint's next epoch; optimizer & LR scheduler are re-created "
              "(checkpoint does not store them to save memory).")

    # 打印配置摘要
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Data source:    {labeled_root}")
    print(f"  Backbone:       {backbone}")
    print(f"  Detection head: {detection_head}")
    print(f"  Image size:     {config['image_size']}")
    print(f"  Batch size:     {config['batch_size']}")
    print(f"  Max epochs:     {config['max_epochs']}")
    print(f"  Learning rate:  {config['learning_rate']}")
    print(f"  Augmentation:   {config['augmentation']}")
    print(f"  Detection:      {config['detection_enabled']}")
    print(f"  Device:         {config.get('device', 'auto')}")
    print(f"  Train samples:  {len(train_samples)}")
    print(f"  Val samples:    {len(val_samples)}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # 开始训练
    history = trainer.fit()

    # 保存训练历史
    import json
    history_path = OUTPUT_DIR / "training_history_labeled.json"
    serializable_history = _make_serializable(history)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2, default=str, ensure_ascii=False)
    print(f"History saved to {history_path}")

    # 最终验证报告
    print("\n" + "=" * 60)
    print("FINAL VALIDATION REPORT")
    print(trainer.val_metrics.get_classification_report())


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()