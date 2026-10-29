"""
离线数据增强：对少数类样本生成增强副本，解决类别不平衡问题。

仅使用颜色/纹理增强（不改变空间布局），因此标注文件可直接复用，
无需变换边界框坐标。

用法:
    python -m data.augment_minority                           # 默认: contact_dermatitis ×10
    python -m data.augment_minority --class contact_dermatitis --factor 10
    python -m data.augment_minority --class all --factor 5    # 增强所有少数类
    python -m data.augment_minority --class urticaria --factor 3
"""

import argparse
import pickle
import random
import shutil
from collections import Counter
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import albumentations as A

from config import OUTPUT_DIR, DATA_ROOT, DISEASE_CLASSES


def create_augmentation_pipeline() -> A.Compose:
    """创建仅含颜色/纹理的增强流水线（不改变空间布局）。

    这样标注文件（JSON/XML）中的 bbox 坐标无需变换，
    可直接复制复用。
    """
    return A.Compose([
        # 亮度/对比度
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.8,
        ),
        # 色调/饱和度/明度
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.6,
        ),
        # 高斯噪声
        A.GaussNoise(var_limit=(5.0, 35.0), p=0.5),
        # 高斯模糊
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        # CLAHE 直方图均衡
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
        # 锐化
        A.Sharpen(alpha=(0.2, 0.5), p=0.3),
        # Gamma 校正
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # 颜色抖动
        A.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03, p=0.5,
        ),
    ])


def augment_minority_class(
    splits: dict,
    target_class: str,
    factor: int = 10,
    seed: int = 42,
) -> dict:
    """对训练集中指定类别的每个样本生成 factor 份增强副本。

    参数:
        splits: train/val/test 划分字典。
        target_class: 要增强的类别名称。
        factor: 每个样本生成的副本数。
        seed: 随机种子。

    返回:
        更新后的 splits 字典。
    """
    random.seed(seed)
    np.random.seed(seed)

    train_samples = splits["train"]
    target_samples = [s for s in train_samples if s["disease"] == target_class]

    if not target_samples:
        print(f"  [WARN] 训练集中没有 '{target_class}' 样本，跳过。")
        return splits

    total_new = len(target_samples) * factor
    print(f"  增强 '{target_class}': {len(target_samples)} 样本 × {factor} = {total_new} 份新样本")

    pipeline = create_augmentation_pipeline()
    new_samples = []
    skipped = 0

    for i, sample in enumerate(target_samples):
        # 解析图像路径
        image_path = sample["image_path"]
        if not Path(image_path).is_absolute():
            image_path = str(DATA_ROOT / image_path)
        image_path = Path(image_path)

        # 加载图像
        img = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if img is None:
            print(f"    [WARN] 无法加载: {image_path}")
            skipped += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 解析标注文件路径
        json_path = None
        xml_path = None
        if sample.get("json_path"):
            p = Path(sample["json_path"])
            json_path = p if p.is_absolute() else DATA_ROOT / p
        if sample.get("xml_path"):
            p = Path(sample["xml_path"])
            xml_path = p if p.is_absolute() else DATA_ROOT / p

        for aug_idx in range(factor):
            # 随机增强
            augmented = pipeline(image=img)
            aug_img = augmented["image"]  # RGB

            # 保存增强图像
            aug_name = f"{image_path.stem}_aug{aug_idx}{image_path.suffix}"
            aug_path = image_path.parent / aug_name
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(image_path.suffix, aug_img_bgr)
            if not success:
                print(f"    [WARN] 编码失败: {aug_name}")
                skipped += 1
                continue
            encoded.tofile(str(aug_path))

            # 相对于 DATA_ROOT 的路径
            try:
                rel_image = str(aug_path.relative_to(DATA_ROOT))
            except ValueError:
                rel_image = str(aug_path)

            # 复制标注文件（bbox 坐标不变，因为只用了颜色增强）
            rel_json = None
            if json_path and json_path.exists():
                aug_json = json_path.parent / f"{json_path.stem}_aug{aug_idx}{json_path.suffix}"
                shutil.copy2(str(json_path), str(aug_json))
                try:
                    rel_json = str(aug_json.relative_to(DATA_ROOT))
                except ValueError:
                    rel_json = str(aug_json)

            rel_xml = None
            if xml_path and xml_path.exists():
                aug_xml = xml_path.parent / f"{xml_path.stem}_aug{aug_idx}{xml_path.suffix}"
                shutil.copy2(str(xml_path), str(aug_xml))
                try:
                    rel_xml = str(aug_xml.relative_to(DATA_ROOT))
                except ValueError:
                    rel_xml = str(aug_xml)

            new_sample = {
                "image_path": rel_image,
                "json_path": rel_json,
                "xml_path": rel_xml,
                "disease": sample["disease"],
                "patient_id": f"{sample['patient_id']}_aug",
            }
            new_samples.append(new_sample)

    if skipped:
        print(f"    [WARN] {skipped} 份增强副本生成失败（已跳过）")

    splits["train"] = train_samples + new_samples
    print(f"    已添加 {len(new_samples)} 份样本。训练集总数: {len(splits['train'])}")
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="离线增强少数类样本以解决类别不平衡",
    )
    parser.add_argument(
        "--class", dest="target_class", type=str, default="contact_dermatitis",
        choices=["contact_dermatitis", "eczema", "bullous_pemphigoid", "urticaria", "all"],
        help="要增强的类别（默认: contact_dermatitis）",
    )
    parser.add_argument(
        "--factor", type=int, default=10,
        help="每个样本生成的增强副本数（默认: 10）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认: 42）",
    )
    args = parser.parse_args()

    # 加载现有划分
    splits_path = OUTPUT_DIR / "splits.pkl"
    if not splits_path.exists():
        print(f"错误: 未找到 splits 文件: {splits_path}")
        print("请先运行 train.py 以生成数据集划分。")
        return

    print(f"加载 splits: {splits_path}")
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    # 打印原始分布
    train_counts = Counter(s["disease"] for s in splits["train"])
    print(f"原始训练集分布: {dict(train_counts)}")

    # 备份原始 splits
    backup_path = splits_path.with_suffix(".pkl.bak")
    print(f"备份原始 splits → {backup_path}")
    shutil.copy2(str(splits_path), str(backup_path))

    # 确定要增强的类别
    if args.target_class == "all":
        # 找出所有样本数少于最大类 50% 的类别
        max_count = max(train_counts.values())
        minority_classes = [
            cls for cls in DISEASE_CLASSES
            if train_counts.get(cls, 0) < max_count * 0.5
        ]
        print(f"\n少数类（< {max_count * 0.5:.0f} 样本）: {minority_classes}")
        for cls_name in minority_classes:
            splits = augment_minority_class(
                splits, cls_name, args.factor, args.seed,
            )
    else:
        splits = augment_minority_class(
            splits, args.target_class, args.factor, args.seed,
        )

    # 保存更新后的 splits
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    # 打印新分布
    new_train_counts = Counter(s["disease"] for s in splits["train"])
    print(f"\n更新后训练集分布: {dict(new_train_counts)}")
    print(f"更新后的 splits 已保存至: {splits_path}")


if __name__ == "__main__":
    main()