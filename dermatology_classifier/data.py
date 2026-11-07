"""
纯疾病分类数据模块。

扫描 labeled/ 与 unlabeled/，按图片所在目录派生疾病标签，
按文件名 stem 去重后合并（union），再做患者级分层划分。
完全不读取 .json 标注文件。
"""

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from config import DISEASE_CN_TO_EN, DISEASE_TO_IDX, DISEASE_CLASSES

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def extract_patient_id(filename: str) -> str:
    """从文件名提取患者 ID，用于患者级分层划分。

    例如:
      "103_0287.JPG" -> "103"
      "_DSC2783.JPG" -> "2783"
      "201202016a.jpg" -> "201202016"
      "治疗后4.jpg" -> "治疗后4"
    """
    stem = Path(filename).stem
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
    # 无法提取可靠数字 ID 时，用完整文件名作为伪 ID，
    # 避免把大量不同图片误归并到同一个"患者"（否则会导致划分严重失衡）。
    return stem


def scan_images(root) -> list[dict]:
    """扫描一个数据根目录（含 湿疹/类天疱疹/荨麻疹 三个子目录）。

    返回样本列表，每个样本 dict 含 image_path / disease / patient_id / source。
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")

    samples = []
    for disease_dir in sorted(root.iterdir()):
        if not disease_dir.is_dir():
            continue
        disease = DISEASE_CN_TO_EN.get(disease_dir.name)
        if disease is None:
            print(f"  [跳过] 未知疾病目录: {disease_dir.name}")
            continue
        for f in sorted(disease_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            samples.append({
                "image_path": str(f),
                "disease": disease,
                "patient_id": extract_patient_id(f.name),
                "source": root.name,
            })
    return samples


def build_dataset(labeled_root, unlabeled_root) -> list[dict]:
    """扫描 labeled + unlabeled，按 (disease, stem) 去重后合并（union），
    再用内容哈希去掉字节级重复的图片（如带 _2 后缀的副本）。"""
    raw = scan_images(labeled_root) + scan_images(unlabeled_root)

    # 1) 按 (disease, stem) 去重：labeled 与 unlabeled 同名图片只保留一份
    seen: set = set()
    dedup: list[dict] = []
    for s in raw:
        key = (s["disease"], Path(s["image_path"]).stem)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)

    # 2) 按内容 MD5 去重：去掉字节级重复（如 _DSC2783_2.JPG 是 _DSC2783.JPG 的副本）
    content_seen: set = set()
    final: list[dict] = []
    n_dup = 0
    for s in dedup:
        try:
            with open(s["image_path"], "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
        except OSError:
            h = None
        if h is not None and h in content_seen:
            n_dup += 1
            continue
        if h is not None:
            content_seen.add(h)
        final.append(s)
    if n_dup:
        print(f"  [去重] 移除 {n_dup} 张内容重复的图片")
    return final


def patient_stratified_split(
    samples: list[dict],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """按 (disease, patient_id) 分组的患者级分层划分。

    同一患者的所有图片不会同时出现在 train / val / test 中。
    """
    patient_groups = defaultdict(list)
    for s in samples:
        patient_groups[(s["disease"], s["patient_id"])].append(s)

    patient_keys = list(patient_groups.keys())
    y = [DISEASE_TO_IDX[k[0]] for k in patient_keys]

    label_counts = Counter(k[0] for k in patient_keys)
    stratify = y if min(label_counts.values()) >= 2 else None
    if stratify is None:
        print("  [提示] 某些类别患者数 < 2，改用随机划分")

    if test_size is not None and test_size <= 0.0:
        train_val_keys, test_keys = patient_keys, []
    else:
        train_val_keys, test_keys = train_test_split(
            patient_keys, test_size=test_size, stratify=stratify,
            random_state=seed,
        )

    tv_y = [DISEASE_TO_IDX[k[0]] for k in train_val_keys]
    val_ratio = val_size / (1.0 - test_size) if test_size > 0 else val_size
    stratify_tv = tv_y if min(label_counts.values()) >= 2 else None
    train_keys, val_keys = train_test_split(
        train_val_keys, test_size=val_ratio, stratify=stratify_tv,
        random_state=seed,
    )

    train_p = set(train_keys)
    val_p = set(val_keys)
    test_p = set(test_keys)
    return {
        "train": [s for s in samples if (s["disease"], s["patient_id"]) in train_p],
        "val": [s for s in samples if (s["disease"], s["patient_id"]) in val_p],
        "test": [s for s in samples if (s["disease"], s["patient_id"]) in test_p],
    }


class DiseaseDataset(Dataset):
    """纯疾病分类数据集，只读图片 + 目录派生疾病标签。"""

    def __init__(self, samples: list[dict], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load(path: str) -> np.ndarray:
        # np.fromfile + cv2.imdecode 兼容含特殊字符（如 '!'）的路径
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = self._load(s["image_path"])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return {
            "image": image,
            "label": DISEASE_TO_IDX[s["disease"]],
            "image_id": Path(s["image_path"]).stem,
            "patient_id": s["patient_id"],
        }


def print_split_stats(splits: dict) -> None:
    """打印各划分的图片数与类别分布。"""
    for name in ("train", "val", "test"):
        ss = splits.get(name, [])
        if not ss:
            continue
        counts = Counter(s["disease"] for s in ss)
        patients = {d: len({s["patient_id"] for s in ss if s["disease"] == d})
                    for d in DISEASE_CLASSES}
        print(f"  {name:5s}: {len(ss)} 张 | " + ", ".join(
            f"{d}={counts.get(d, 0)}" for d in DISEASE_CLASSES))
        print(f"          患者数: " + ", ".join(
            f"{d}={patients.get(d, 0)}" for d in DISEASE_CLASSES))
