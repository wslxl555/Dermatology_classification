"""
数据集准备:扫描数据目录,构建样本列表,
并创建分层的 train/val/test 划分。

运行方式:python -m data.prepare_data
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional

from sklearn.model_selection import StratifiedKFold, train_test_split

from config import DATA_ROOT, DISEASE_DIR_MAP, DISEASE_CLASSES, TRAINING_CONFIG


def _is_image(filename: str) -> bool:
    """判断文件名是否为图像文件。"""
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


def _is_json(filename: str) -> bool:
    """判断文件名是否为 JSON 标注文件。"""
    return filename.lower().endswith(".json")


def _is_xml(filename: str) -> bool:
    """判断文件名是否为 XML 标注文件。"""
    return filename.lower().endswith(".xml")


def _find_annotation(image_path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    为一张图像查找对应的标注文件。
    返回 (json_path, xml_path) —— 两者可能都为 None。
    """
    stem = image_path.stem
    parent = image_path.parent

    json_path = parent / (stem + ".json")
    xml_path = parent / (stem + ".xml")

    json_str = str(json_path) if json_path.exists() else None
    xml_str = str(xml_path) if xml_path.exists() else None

    return json_str, xml_str


def _extract_position(parent_dir_name: str) -> str:
    """
    从父目录名提取拍摄位置代码 (L/S/J/B)。

    例如:
        "0001 湿疹L" → "L"
        "0002 荨麻疹 S" → "S"
        "0003 天疱疮 J" → "J"
        "0004 荨麻疹 B" → "B"
    如果无法识别则返回 "unknown"。
    """
    name = parent_dir_name.strip()
    # 先尝试空格分隔: "0002 荨麻疹 S" → ["0002", "荨麻疹", "S"]
    parts = name.split()
    if len(parts) >= 2:
        last_token = parts[-1].strip()
        if last_token in ("L", "S", "J", "B"):
            return last_token
    # 无空格情况: "0001 湿疹L" → 以 L/S/J/B 结尾
    for pos in ("L", "S", "J", "B"):
        if name.endswith(pos):
            return pos
    return "unknown"


def _extract_patient_id(rel_path: str, is_labelme: bool = False) -> str:
    """
    从相对路径中提取患者 ID。
    例如 "0001 湿疹L/103_0570.JPG" → "0001"
    例如 "0004 荨麻疹/0001 荨麻疹 L/img.JPG" → "0004"

    对于 labelme 扁平目录中的文件（无患者子目录），
    尝试从文件名前缀提取患者 ID。
    """
    parts = rel_path.split(os.sep)

    # 首先检查目录名中是否有患者 ID（常规患者子目录）
    for part in parts[:-1]:  # 排除文件名本身
        part = part.strip()
        if len(part) >= 4 and part[:4].isdigit():
            return part.split()[0] if ' ' in part else part[:4]

    # 对于 labelme 文件（扁平目录，无患者子目录），尝试从文件名提取
    if is_labelme:
        filename = parts[-1] if parts else ""
        # 模式 1: "103_0570.JPG" → 患者 "103"
        # 模式 2: "_DSC2783.JPG" → 无法提取，使用文件名前缀
        prefix = filename.split("_")[0] if "_" in filename else filename[:4]
        if prefix and prefix[0].isdigit() and len(prefix) >= 3:
            return f"labelme_{prefix}"
        else:
            # 无法提取，使用文件名前 8 字符作为伪 ID
            return f"labelme_{filename[:8]}"

    return "unknown"


def scan_dataset(data_root: Path = DATA_ROOT, data_tier: str = "入选") -> list[dict]:
    """
    扫描整个数据集并构建样本 dict 列表。

    参数:
        data_root: 数据集根目录。
        data_tier: "入选" (仅高质量数据) 或 "all" (全部数据)。

    每个样本: {
        'image_path': str,
        'json_path': str 或 None,
        'xml_path': str 或 None,
        'disease': str,       # DISEASE_CLASSES 之一
        'patient_id': str,
        'data_quality': str,  # "入选" | "不完整L+S" | "不完整仅有L" | "labelme" | ""
        'position': str,      # "L" | "S" | "J" | "B" | "unknown"
    }
    """
    from config import DATA_TIER_CONFIG

    tier_cfg = DATA_TIER_CONFIG.get(data_tier, DATA_TIER_CONFIG["入选"])
    samples = []

    # --- 湿疹 (eczema) ---
    eczema_dir = DISEASE_DIR_MAP["eczema"]
    if eczema_dir.exists():
        for subdir_name in tier_cfg["eczema_subdirs"]:
            subdir = eczema_dir / subdir_name
            if not subdir.exists():
                print(f"  [WARN] Eczema subdir not found: {subdir}")
                continue

            is_labelme = (subdir_name == "labelme已标注数据")
            quality = "labelme" if is_labelme else "入选"

            if is_labelme:
                print(f"  Scanning eczema ({quality}): {subdir}")
                for f in sorted(subdir.iterdir()):
                    if not _is_image(f.name):
                        continue
                    image_path = subdir / f.name
                    json_path, xml_path = _find_annotation(image_path)
                    try:
                        rel_path = str(image_path.relative_to(eczema_dir))
                    except ValueError:
                        rel_path = str(image_path)
                    patient_id = _extract_patient_id(rel_path, is_labelme=True)
                    try:
                        rel_image = str(image_path.relative_to(DATA_ROOT))
                    except ValueError:
                        rel_image = str(image_path)
                    rel_json = _rel_path(json_path, DATA_ROOT)
                    rel_xml = _rel_path(xml_path, DATA_ROOT)
                    samples.append({
                        "image_path": rel_image,
                        "json_path": rel_json,
                        "xml_path": rel_xml,
                        "disease": "eczema",
                        "patient_id": patient_id,
                        "data_quality": quality,
                        "position": "unknown",  # labelme 扁平目录无位置信息
                    })
            else:
                print(f"  Scanning eczema ({quality}): {subdir}")
                for patient_dir in sorted(subdir.iterdir()):
                    if not patient_dir.is_dir():
                        continue
                    for root, dirs, files in os.walk(patient_dir):
                        for f in files:
                            if not _is_image(f):
                                continue
                            image_path = Path(root) / f
                            json_path, xml_path = _find_annotation(image_path)
                            try:
                                rel_path = str(image_path.relative_to(eczema_dir))
                            except ValueError:
                                rel_path = str(image_path)
                            patient_id = _extract_patient_id(rel_path)
                            position = _extract_position(Path(root).name)
                            try:
                                rel_image = str(image_path.relative_to(DATA_ROOT))
                            except ValueError:
                                rel_image = str(image_path)
                            rel_json = _rel_path(json_path, DATA_ROOT)
                            rel_xml = _rel_path(xml_path, DATA_ROOT)
                            samples.append({
                                "image_path": rel_image,
                                "json_path": rel_json,
                                "xml_path": rel_xml,
                                "disease": "eczema",
                                "patient_id": patient_id,
                                "data_quality": quality,
                                "position": position,
                            })
    else:
        print(f"  [WARN] Eczema directory not found: {eczema_dir}")

    # --- 大疱型类天疱疮 (bullous_pemphigoid) ---
    tpc_dir = DISEASE_DIR_MAP["bullous_pemphigoid"]
    if tpc_dir.exists():
        print(f"  Scanning bullous_pemphigoid: {tpc_dir}")
        for patient_dir in sorted(tpc_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            for root, dirs, files in os.walk(patient_dir):
                for f in files:
                    if not _is_image(f):
                        continue
                    image_path = Path(root) / f
                    json_path, xml_path = _find_annotation(image_path)
                    try:
                        rel_path = str(image_path.relative_to(tpc_dir))
                    except ValueError:
                        rel_path = str(image_path)
                    patient_id = _extract_patient_id(rel_path)
                    position = _extract_position(Path(root).name)
                    try:
                        rel_image = str(image_path.relative_to(DATA_ROOT))
                    except ValueError:
                        rel_image = str(image_path)
                    rel_json = _rel_path(json_path, DATA_ROOT)
                    rel_xml = _rel_path(xml_path, DATA_ROOT)
                    samples.append({
                        "image_path": rel_image,
                        "json_path": rel_json,
                        "xml_path": rel_xml,
                        "disease": "bullous_pemphigoid",
                        "patient_id": patient_id,
                        "data_quality": "入选",
                        "position": position,
                    })
    else:
        print(f"  [WARN] Bullous pemphigoid directory not found: {tpc_dir}")

    # --- 荨麻疹 (urticaria) ---
    xmz_dir = DISEASE_DIR_MAP["urticaria"]
    if xmz_dir.exists():
        for subdir_name in tier_cfg["urticaria_subdirs"]:
            subdir = xmz_dir / subdir_name
            if not subdir.exists():
                print(f"  [WARN] Urticaria subdir not found: {subdir}")
                continue

            if "完整数据" in subdir_name:
                quality = "入选"
            elif "仅有L" in subdir_name:
                quality = "不完整仅有L"
            else:
                quality = "不完整L+S"

            print(f"  Scanning urticaria ({quality}): {subdir}")
            for patient_dir in sorted(subdir.iterdir()):
                if not patient_dir.is_dir():
                    continue
                for root, dirs, files in os.walk(patient_dir):
                    for f in files:
                        if not _is_image(f):
                            continue
                        image_path = Path(root) / f
                        json_path, xml_path = _find_annotation(image_path)
                        try:
                            rel_path = str(image_path.relative_to(xmz_dir))
                        except ValueError:
                            rel_path = str(image_path)
                        patient_id = _extract_patient_id(rel_path)
                        position = _extract_position(Path(root).name)
                        try:
                            rel_image = str(image_path.relative_to(DATA_ROOT))
                        except ValueError:
                            rel_image = str(image_path)
                        rel_json = _rel_path(json_path, DATA_ROOT)
                        rel_xml = _rel_path(xml_path, DATA_ROOT)
                        samples.append({
                            "image_path": rel_image,
                            "json_path": rel_json,
                            "xml_path": rel_xml,
                            "disease": "urticaria",
                            "patient_id": patient_id,
                            "data_quality": quality,
                            "position": position,
                        })
    else:
        print(f"  [WARN] Urticaria directory not found: {xmz_dir}")

    return samples


def _rel_path(path_str: Optional[str], root: Path) -> Optional[str]:
    """将绝对路径转为相对路径（相对于 root）。"""
    if path_str is None:
        return None
    try:
        return str(Path(path_str).relative_to(root))
    except ValueError:
        return path_str


def create_splits(
    samples: list[dict],
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 42,
    use_kfold: bool = False,
    n_folds: int = 5,
    min_val_per_class: int = 2,
    tiny_class_threshold: int = 10,
    tiny_class_val_ratio: float = 0.30,
) -> dict:
    """
    创建 train/val/test 划分。

    如果 use_kfold=True,返回 5 折交叉验证划分。
    否则返回单一的 train/val/test 划分。

    划分按疾病 + 患者进行分层(同一患者不会同时
    出现在 train 和 val/test 中)。

    对于患者数少于 tiny_class_threshold 的小类别,
    使用 tiny_class_val_ratio 作为 val 分配比例,
    并确保 val 中至少有 min_val_per_class 个患者。
    这避免了小类别在 val 中只有 1-2 个样本的问题。
    """
    rng = random.Random(seed)

    # 按 (疾病, 患者 ID) 对样本分组
    patient_groups = defaultdict(list)
    for sample in samples:
        key = (sample["disease"], sample["patient_id"])
        patient_groups[key].append(sample)

    patient_keys = list(patient_groups.keys())
    patient_labels = [key[0] for key in patient_keys]  # 疾病名称

    # 创建疾病到索引的映射用于分层
    disease_to_idx = {d: i for i, d in enumerate(DISEASE_CLASSES)}
    y = [disease_to_idx[label] for label in patient_labels]

    if use_kfold:
        # 5 折交叉验证
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        for train_idx, val_idx in skf.split(patient_keys, y):
            train_patients = set(patient_keys[i] for i in train_idx)
            val_patients = set(patient_keys[i] for i in val_idx)

            train_samples = [s for s in samples
                             if (s["disease"], s["patient_id"]) in train_patients]
            val_samples = [s for s in samples
                           if (s["disease"], s["patient_id"]) in val_patients]

            folds.append({"train": train_samples, "val": val_samples})
        return {"folds": folds, "n_folds": n_folds}
    else:
        # 单一 train/val/test 划分
        # 第一步: train+val vs test (全局分层)
        train_val_keys, test_keys = train_test_split(
            patient_keys, test_size=test_size,
            stratify=y, random_state=seed,
        )

        # 第二步: per-class train vs val 划分
        # 按疾病类别分组 train_val 中的患者
        train_val_by_class = defaultdict(list)
        for key in train_val_keys:
            train_val_by_class[key[0]].append(key)

        train_keys_split = []
        val_keys_split = []

        for disease, keys in train_val_by_class.items():
            n_patients = len(keys)
            if n_patients <= 1:
                # 只有 1 个患者:全部放入 train
                train_keys_split.extend(keys)
                continue

            # 确定 val 患者数量
            if n_patients < tiny_class_threshold:
                # 小类别:使用更高的 val 比例,确保至少 min_val_per_class 个患者
                n_val_patients = max(min_val_per_class,
                                     int(__import__("math").ceil(n_patients * tiny_class_val_ratio)))
            else:
                n_val_patients = max(1, int(round(n_patients * val_size)))

            # 确保 train 至少保留 1 个患者
            n_val_patients = min(n_val_patients, n_patients - 1)

            shuffled = list(keys)
            rng.shuffle(shuffled)
            val_keys_split.extend(shuffled[:n_val_patients])
            train_keys_split.extend(shuffled[n_val_patients:])

        train_patients = set(train_keys_split)
        val_patients = set(val_keys_split)
        test_patients = set(test_keys)

        train_samples = [s for s in samples
                         if (s["disease"], s["patient_id"]) in train_patients]
        val_samples = [s for s in samples
                       if (s["disease"], s["patient_id"]) in val_patients]
        test_samples = [s for s in samples
                        if (s["disease"], s["patient_id"]) in test_patients]

        return {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }


def print_statistics(samples: list[dict], splits: dict = None) -> None:
    """打印数据集统计信息。"""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # 总体统计
    disease_counts = defaultdict(int)
    annotation_counts = defaultdict(int)
    quality_counts = defaultdict(lambda: defaultdict(int))
    for s in samples:
        disease_counts[s["disease"]] += 1
        if s["json_path"] or s["xml_path"]:
            annotation_counts[s["disease"]] += 1
        quality = s.get("data_quality", "")
        quality_counts[s["disease"]][quality] += 1

    print(f"\nTotal samples: {len(samples)}")
    print(f"\nPer disease:")
    for disease in DISEASE_CLASSES:
        n = disease_counts.get(disease, 0)
        n_annot = annotation_counts.get(disease, 0)
        qinfo = ", ".join(f"{q}={c}" for q, c in sorted(quality_counts[disease].items()))
        print(f"  {disease}: {n} images, {n_annot} with annotations  [{qinfo}]")

    # 划分统计
    if splits:
        if "folds" in splits:
            print(f"\n{len(splits['folds'])}-fold CV")
            for i, fold in enumerate(splits["folds"]):
                print(f"  Fold {i}: train={len(fold['train'])}, val={len(fold['val'])}")
        else:
            print(f"\nSplits:")
            for split_name in ["train", "val", "test"]:
                if split_name in splits:
                    split_samples = splits[split_name]
                    split_counts = defaultdict(int)
                    split_patients = defaultdict(set)
                    for s in split_samples:
                        split_counts[s["disease"]] += 1
                        split_patients[s["disease"]].add(s["patient_id"])
                    print(f"  {split_name}: {len(split_samples)} total")
                    for disease in DISEASE_CLASSES:
                        n_img = split_counts.get(disease, 0)
                        n_pat = len(split_patients.get(disease, set()))
                        print(f"    {disease}: {n_img} images, {n_pat} patients")

    # 皮损标签多样性
    print(f"\nLesion label statistics: see label_mapping.diagnose_unmapped_labels()")

    # 位置分布
    from collections import Counter
    position_counts = Counter(s.get("position", "unknown") for s in samples)
    print(f"\nPosition distribution:")
    for pos in ["L", "S", "J", "B", "unknown"]:
        c = position_counts.get(pos, 0)
        if c > 0:
            print(f"  {pos}: {c}")


def main(data_tier: str = "入选"):
    print(f"Scanning dataset (data_tier={data_tier})...")
    samples = scan_dataset(data_tier=data_tier)

    if not samples:
        print("ERROR: No samples found!")
        return

    print(f"\nFound {len(samples)} total samples.")

    # 创建划分
    print("\nCreating train/val/test splits (patient-level stratified, "
          "per-class min val constraint)...")
    splits = create_splits(
        samples,
        val_size=TRAINING_CONFIG["val_split"],
        test_size=0.1,
        seed=TRAINING_CONFIG["seed"],
        use_kfold=False,
        min_val_per_class=2,
        tiny_class_threshold=10,
        tiny_class_val_ratio=0.30,
    )

    print_statistics(samples, splits)

    # 将划分保存为 pickle 以便复现
    import pickle
    save_path = Path(__file__).resolve().parent.parent / "outputs" / "splits.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"\nSplits saved to: {save_path}")

    # 诊断未映射的标签
    from data.label_mapping import diagnose_unmapped_labels
    unmapped = diagnose_unmapped_labels(str(DATA_ROOT))
    if unmapped:
        print(f"\nUnmapped labels ({len(unmapped)}):")
        for label, count in list(unmapped.items())[:20]:
            print(f"  [{count:4d}] '{label}'")
    else:
        print("\nAll labels mapped successfully!")


if __name__ == "__main__":
    import sys
    tier = sys.argv[1] if len(sys.argv) > 1 else "入选"
    if tier not in ("入选", "all"):
        print(f"Usage: python -m data.prepare_data [入选|all]")
        sys.exit(1)
    main(data_tier=tier)
