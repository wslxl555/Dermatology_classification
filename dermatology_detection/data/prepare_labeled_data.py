"""
从 labeled/ 文件夹扫描已标注数据，构建样本列表和数据集划分。

目录结构:
  labeled/{疾病中文名}/
    ├── image1.jpg / .JPG / .png
    ├── image1.json          # LabelMe 格式标注
    ├── image2.jpg
    └── image2.json

运行方式: python -m data.prepare_labeled_data
"""

import os
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional

from sklearn.model_selection import train_test_split

# 疾病中文名 → 英文映射
DISEASE_CN_TO_EN = {
    "湿疹": "eczema",
    "类天疱疹": "bullous_pemphigoid",
    "荨麻疹": "urticaria",
}

# 英文 → 索引
DISEASE_TO_IDX = {
    "eczema": 0,
    "bullous_pemphigoid": 1,
    "urticaria": 2,
}

DISEASE_CLASSES = list(DISEASE_TO_IDX.keys())


def _is_image(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


def _is_json(filename: str) -> bool:
    return filename.lower().endswith(".json")


def _extract_patient_id(filename: str) -> str:
    """从文件名提取患者ID，用于分层划分。
    例如:
      "103_0287.JPG" → "103"
      "DSC03345.JPG" → "DSC03345"
      "1106058a.JPG" → "1106058"
      "201202016a.jpg" → "201202016"
      "治疗后4.jpg" → "治疗后4"
      "733660067815768316.jpg" → "733660067815768316"
    """
    stem = Path(filename).stem
    # 尝试提取数字前缀
    digits = ""
    for ch in stem:
        if ch.isdigit():
            digits += ch
        elif digits and len(digits) >= 3:
            break
        else:
            digits = ""
    if len(digits) >= 3:
        return digits
    # 使用文件名前8个字符作为伪ID
    return stem[:8] if len(stem) >= 8 else stem


def scan_labeled_dataset(
    labeled_root: str | Path = None
) -> list[dict]:
    """
    扫描 labeled/ 文件夹，构建样本列表。

    参数:
        labeled_root: labeled 文件夹路径，默认使用项目根目录下的 labeled/

    返回:
        样本 dict 列表，每个样本包含:
          - image_path: str (绝对路径)
          - json_path: str 或 None (绝对路径)
          - disease: str (英文疾病名)
          - patient_id: str (用于分层划分)
          - data_quality: str ("labeled")
          - position: str ("unknown")
    """
    if labeled_root is None:
        labeled_root = Path(__file__).resolve().parent.parent.parent / "labeled"
    labeled_root = Path(labeled_root)

    if not labeled_root.exists():
        raise FileNotFoundError(f"Labeled directory not found: {labeled_root}")

    samples = []
    total_images = 0
    total_json = 0

    for disease_dir in sorted(labeled_root.iterdir()):
        if not disease_dir.is_dir():
            continue

        disease_cn = disease_dir.name
        disease_en = DISEASE_CN_TO_EN.get(disease_cn)
        if disease_en is None:
            print(f"  [SKIP] Unknown disease directory: {disease_cn}")
            continue

        print(f"Scanning {disease_cn} ({disease_en}): {disease_dir}")

        # 收集该目录下的所有图片和 JSON 文件
        images = {}
        jsons = set()
        for f in disease_dir.iterdir():
            if _is_image(f.name):
                images[f.stem] = str(f)
            elif _is_json(f.name):
                jsons.add(f.stem)

        n_images = len(images)
        n_json = len(jsons)
        print(f"  Found {n_images} images, {n_json} JSON files")
        total_images += n_images
        total_json += n_json

        # 配对图片和标注
        paired = 0
        for stem, image_path in sorted(images.items()):
            json_path = None
            # 尝试精确匹配 stem
            if stem in jsons:
                json_path = str(disease_dir / (stem + ".json"))
            else:
                # 尝试查找以 stem 开头的 JSON 文件（处理命名不一致）
                for json_stem in jsons:
                    if json_stem.startswith(stem) or stem.startswith(json_stem):
                        json_path = str(disease_dir / (json_stem + ".json"))
                        break

            patient_id = _extract_patient_id(Path(image_path).name)

            samples.append({
                "image_path": image_path,
                "json_path": json_path,
                "disease": disease_en,
                "patient_id": patient_id,
                "data_quality": "labeled",
                "position": "unknown",  # labeled 数据无位置信息
            })

            if json_path:
                paired += 1

        print(f"  Paired: {paired}/{n_images}")

    print(f"\nTotal: {len(samples)} samples across {total_images} images")
    return samples


def create_labeled_splits(
    samples: list[dict],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    创建按患者分层的 train/val/test 划分。

    同一患者的所有图片不会同时出现在 train 和 val/test 中。

    返回: {"train": [...], "val": [...], "test": [...]}
    """
    rng = random.Random(seed)

    # 按 (disease, patient_id) 分组
    patient_groups = defaultdict(list)
    for s in samples:
        key = (s["disease"], s["patient_id"])
        patient_groups[key].append(s)

    patient_keys = list(patient_groups.keys())
    patient_labels = [key[0] for key in patient_keys]
    y = [DISEASE_TO_IDX[label] for label in patient_labels]

    # 第一步: train+val vs test
    # 检查每个类别的患者数，确保分层划分可行
    label_counts = defaultdict(int)
    for label in patient_labels:
        label_counts[label] += 1

    # 对于类别数 < 2 个患者的，使用简单随机划分
    stratify_y = y if min(label_counts.values()) >= 2 else None
    if stratify_y is None:
        print("  [WARN] Some classes have < 2 patients, using random split")

    if test_size is not None and test_size <= 0.0:
        # 不预留 test 集（例如测试集改用 unlabeled/），全部患者进入 train/val。
        # 注：sklearn 的 train_test_split 不接受 test_size=0.0，需在此单独处理。
        train_val_keys = patient_keys
        test_keys = []
    else:
        train_val_keys, test_keys = train_test_split(
            patient_keys,
            test_size=test_size,
            stratify=stratify_y,
            random_state=seed,
        )

    # 第二步: train vs val
    train_val_labels = [key[0] for key in train_val_keys]
    train_val_y = [DISEASE_TO_IDX[l] for l in train_val_labels]
    val_ratio = val_size / (1 - test_size)
    stratify_tv = train_val_y if min(label_counts.values()) >= 2 else None

    train_keys, val_keys = train_test_split(
        train_val_keys,
        test_size=val_ratio,
        stratify=stratify_tv,
        random_state=seed,
    )

    # 构建样本集合
    train_patients = set(train_keys)
    val_patients = set(val_keys)
    test_patients = set(test_keys)

    train_samples = [s for s in samples if (s["disease"], s["patient_id"]) in train_patients]
    val_samples = [s for s in samples if (s["disease"], s["patient_id"]) in val_patients]
    test_samples = [s for s in samples if (s["disease"], s["patient_id"]) in test_patients]

    return {"train": train_samples, "val": val_samples, "test": test_samples}


def print_labeled_statistics(samples: list[dict], splits: dict = None) -> None:
    """打印数据集统计信息。"""
    print("\n" + "=" * 60)
    print("LABELED DATASET STATISTICS")
    print("=" * 60)

    disease_counts = defaultdict(int)
    annotation_counts = defaultdict(int)
    for s in samples:
        disease_counts[s["disease"]] += 1
        if s["json_path"]:
            annotation_counts[s["disease"]] += 1

    print(f"\nTotal samples: {len(samples)}")
    print(f"Per disease:")
    for disease in DISEASE_CLASSES:
        n_img = disease_counts.get(disease, 0)
        n_annot = annotation_counts.get(disease, 0)
        print(f"  {disease}: {n_img} images, {n_annot} with annotations")

    if splits:
        print(f"\nSplits:")
        for split_name in ["train", "val", "test"]:
            if split_name not in splits:
                continue
            split_samples = splits[split_name]
            split_counts = defaultdict(int)
            split_patients = defaultdict(set)
            for s in split_samples:
                split_counts[s["disease"]] += 1
                split_patients[s["disease"]].add(s["patient_id"])
            print(f"  {split_name}: {len(split_samples)} images")
            for disease in DISEASE_CLASSES:
                n_img = split_counts.get(disease, 0)
                n_pat = len(split_patients.get(disease, set()))
                if n_img > 0:
                    print(f"    {disease}: {n_img} images, {n_pat} patients")

    # 读取一些 JSON 来统计标签分布
    print(f"\nLesion label distribution (sampling up to 50 JSONs per disease):")
    from collections import Counter
    label_counter = Counter()
    for disease in DISEASE_CLASSES:
        disease_samples = [s for s in samples if s["disease"] == disease and s["json_path"]]
        for s in disease_samples[:50]:
            try:
                import json
                with open(s["json_path"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    label = shape.get("label", "").strip()
                    if label:
                        label_counter[label] += 1
            except Exception:
                pass
    if label_counter:
        print(f"  Top labels:")
        for label, count in label_counter.most_common(30):
            try:
                print(f"    [{count:4d}] {label}")
            except UnicodeEncodeError:
                # 处理无法打印的字符（如零宽字符）
                safe_label = label.encode("ascii", errors="replace").decode("ascii")
                print(f"    [{count:4d}] {safe_label}")


def main():
    import sys
    labeled_root = sys.argv[1] if len(sys.argv) > 1 else None

    print("Scanning labeled dataset...")
    samples = scan_labeled_dataset(labeled_root)

    if not samples:
        print("ERROR: No samples found!")
        return

    # 创建划分
    print("\nCreating train/val/test splits (patient-level stratified)...")
    splits = create_labeled_splits(samples, val_size=0.15, test_size=0.15, seed=42)

    print_labeled_statistics(samples, splits)

    # 保存划分
    import pickle
    save_dir = Path(__file__).resolve().parent.parent / "outputs"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "labeled_splits.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"\nSplits saved to: {save_path}")


if __name__ == "__main__":
    main()