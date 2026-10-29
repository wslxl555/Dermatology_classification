"""
纯疾病分类数据集。

只读取**图片**与目录派生的**疾病标签**，完全不读取 .json 标注文件。
疾病标签来自样本字典中的 `disease` 字段（由 prepare_labeled_data 的
目录名映射得到），与 LabelMe 标注无关。

因此，验证/推理阶段无需任何 .json 文件，只需要图片即可完成分类。
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from data.prepare_labeled_data import DISEASE_TO_IDX


class DiseaseClassificationDataset(Dataset):
    """
    每个样本返回:
        - image: Tensor [C, H, W]
        - disease_label: int (0-2)
        - image_id: str
    """

    def __init__(
        self,
        samples: list[dict],
        transform=None,
        disease_to_idx: dict | None = None,
    ):
        """
        参数:
            samples: 样本字典列表，每个至少包含 "image_path" 与 "disease"。
                     （"image_path" 为绝对路径，来自 scan_labeled_dataset。）
            transform: Albumentations 增强（不含 bbox_params）。
            disease_to_idx: 疾病名 → 索引 映射。
        """
        self.samples = samples
        self.transform = transform
        self.disease_to_idx = disease_to_idx or DISEASE_TO_IDX

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: str) -> np.ndarray:
        """读取图片为 RGB numpy 数组（用 np.fromfile 处理含特殊字符的路径）。"""
        img = cv2.imdecode(
            np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR,
        )
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # 只读取图片，不读取任何标注文件。
        image = self._load_image(sample["image_path"])

        if self.transform:
            image = self.transform(image=image)["image"]

        # 疾病标签来自目录，而非 .json。
        disease_label = self.disease_to_idx[sample["disease"]]

        return {
            "image": image,
            "disease_label": disease_label,
            "image_id": Path(sample["image_path"]).stem,
        }
