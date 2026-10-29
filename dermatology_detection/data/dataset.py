"""
湿疹皮肤病数据集的 PyTorch Dataset。
加载带有 LabelMe/VOC 标注的图像,返回标准化后的张量。
"""

import os
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATA_ROOT, POSITION_CLASSES
from data.label_mapping import (
    LESION_CLASSES,
    LESION_CLASS_TO_IDX,
    parse_label,
    label_to_multi_hot,
    label_to_indices,
)
from utils.labelme_parser import parse_labelme
from utils.voc_parser import parse_voc_xml


def _resolve_path(path: str | None, root: Path = DATA_ROOT) -> str | None:
    """
    将可能的相对路径基于 DATA_ROOT 解析为绝对路径。

    splits 中保存的路径相对于 DATA_ROOT,这样 splits 文件
    可以跨机器移植。绝对路径原样返回,以便旧的 Windows 绝对路径
    splits.pkl 文件仍然可用(前提是该路径确实存在)。
    """
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return path  # 绝对路径 —— 原样使用(向后兼容)
    return str(root / p)


class EczemaDataset(Dataset):
    """
    用于带皮损标注的皮肤病图像的 PyTorch Dataset。

    每个样本返回:
        - image: Tensor [C, H, W]
        - disease_label: int (0-2)
        - lesion_multi_hot: List[int] (长度 15)
        - bboxes: Tensor [N, 4],格式为 (x1, y1, x2, y2),或为空
        - lesion_labels: Tensor [N],每个 bbox 对应的皮损类别索引,或为空
        - image_id: str
    """

    def __init__(
        self,
        samples: list[dict],
        transform=None,
        lesion_classes: list[str] = LESION_CLASSES,
    ):
        """
        参数:
            samples: 样本字典列表,每个包含:
                - image_path: str(相对 DATA_ROOT,或绝对路径)
                - json_path: Optional[str]
                - xml_path: Optional[str]
                - disease: str("eczema", "contact_dermatitis" 等)
                - patient_id: str
            transform: Albumentations 数据增强(带 bbox_params 的 Compose)
            lesion_classes: 标准化皮损类别名称列表
        """
        self.samples = samples
        self.transform = transform
        self.lesion_classes = lesion_classes
        self.class_to_idx = {name: i for i, name in enumerate(lesion_classes)}

        # 疾病标签映射 (3 类):将字符串类别转换为模型需要的整数索引,
        # 后续在损失函数与分类报告中统一使用这些编号。
        self.disease_to_idx = {
            "eczema": 0,
            "bullous_pemphigoid": 1,
            "urticaria": 2,
        }

        # 位置标签映射 (4 类):L/S/J/B → 0/1/2/3
        self.position_to_idx = {pos: i for i, pos in enumerate(POSITION_CLASSES)}

    def __len__(self) -> int:
        """返回数据集的样本数量。"""
        return len(self.samples)

    def _load_image(self, image_path: str) -> np.ndarray:
        """将图像加载为 RGB 的 numpy 数组。
        使用 np.fromfile + cv2.imdecode 来处理包含特殊字符的路径
        (例如 Windows 路径中的 '!',cv2.imread 无法处理)。
        """
        img = cv2.imdecode(
            np.fromfile(image_path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_annotations(self, sample: dict) -> tuple[list, list[list], list[str]]:
        """
        从 JSON 或 XML 加载标注。
        返回:(bboxes, 每个 bbox 的类别索引列表, 原始标签)
        """
        bboxes = []
        class_indices = []
        all_raw_labels = []

        # 读取 LabelMe 格式标注:每个标注包含原始标签和对应的矩形框。
        # 这些信息在训练阶段会被转成 lesion multi-hot 向量和检测目标。
        json_path = sample.get("json_path")
        if json_path and os.path.exists(json_path):
            parsed = parse_labelme(Path(json_path))
            if parsed and parsed.get("annotations"):
                for ann in parsed["annotations"]:
                    raw = ann["label"]
                    all_raw_labels.append(raw)

                    # bbox 格式为 [x1, y1, x2, y2],表示病灶的边界框。
                    bbox = ann["bbox"]  # [x1, y1, x2, y2]
                    bboxes.append(bbox)

                    # 同一病灶可能包含多个皮损类别(如 "papule, scale"),
                    # 因此需要把多个标签映射为多个类别索引。
                    parsed_classes = parse_label(raw)
                    indices = [self.class_to_idx[c] for c in parsed_classes
                               if c in self.class_to_idx]
                    class_indices.append(indices if indices else [0])  # 默认使用类别 0

        # 读取 VOC XML 标注:与 LabelMe 的处理逻辑一致,整合为统一数据结构。
        xml_path = sample.get("xml_path")
        if xml_path and os.path.exists(xml_path):
            parsed = parse_voc_xml(Path(xml_path))
            if parsed and parsed.get("annotations"):
                for ann in parsed["annotations"]:
                    raw = ann["label"]
                    all_raw_labels.append(raw)

                    bbox = ann["bbox"]
                    bboxes.append(bbox)

                    parsed_classes = parse_label(raw)
                    indices = [self.class_to_idx[c] for c in parsed_classes
                               if c in self.class_to_idx]
                    class_indices.append(indices if indices else [0])

        return bboxes, class_indices, all_raw_labels

    def __getitem__(self, idx: int) -> dict:
        """根据索引读取一个样本,返回图像、标签与检测目标等字典。"""
        sample = self.samples[idx]

        # 路径可能以相对路径保存在 splits 中,需在这里解析为实际文件位置。
        # 这样同一份数据索引可以跨平台/跨机器复用。
        image_path = _resolve_path(sample["image_path"])

        # 读取原始图像并记录原始分辨率,便于后续检测框还原。
        image = self._load_image(image_path)
        orig_h, orig_w = image.shape[:2]

        # 解析 XML/JSON 标注文件路径,并统一成绝对路径以便读取。
        resolved_sample = {
            **sample,
            "image_path": image_path,
            "json_path": _resolve_path(sample.get("json_path")),
            "xml_path": _resolve_path(sample.get("xml_path")),
        }

        # 从标注中读取病灶边界框与对应的原始标签。
        bboxes, class_indices, raw_labels = self._load_annotations(resolved_sample)

        # 疾病标签:因为每张图属于某一种皮肤病,转换为整数索引供分类头使用。
        disease = sample.get("disease", "eczema")
        disease_label = self.disease_to_idx.get(disease, 0)

        # 位置标签:从子目录名称提取的 L/S/J/B 拍摄位置
        position = sample.get("position", "unknown")
        position_label = self.position_to_idx.get(position, -1)  # unknown → -1 (ignore)

        # lesion_multi_hot 用于描述“这张图里出现了哪些皮损类型”,
        # 其本质是多标签分类目标,适合同时存在多种病灶表现的场景。
        lesion_multi_hot = label_to_multi_hot(raw_labels)

        # 训练中应用数据增强;如果存在 bbox,则需要同步更新边界框坐标。
        # 一个框可能带多个皮损类别(如 "Plaque+Scale"),增强阶段先按主类别对齐,
        # 增强完成后再展开成多个 (框, 类别) 对,保留复合标签的检测监督。
        if self.transform and bboxes:
            # albumentations v2 使用像素坐标的 bbox(输入与输出都是像素值)。
            flat_labels = [idx[0] for idx in class_indices]  # 每个 bbox 的主类别
            valid_bboxes = []
            valid_flat_labels = []
            valid_class_indices = []
            for i, (x1, y1, x2, y2) in enumerate(bboxes):
                # 过滤掉面积为 0 或负数的无效框,避免训练阶段出现异常样本。
                if x2 > x1 and y2 > y1:
                    valid_bboxes.append([x1, y1, x2, y2])
                    valid_flat_labels.append(flat_labels[i])
                    valid_class_indices.append(class_indices[i])

            if valid_bboxes:
                transformed = self.transform(
                    image=image,
                    bboxes=valid_bboxes,
                    class_labels=valid_flat_labels,
                )
                image = transformed["image"]
                # albumentations v2 输出即为增强后图像尺寸下的像素坐标 bbox。
                bboxes = [
                    [float(x1), float(y1), float(x2), float(y2)]
                    for (x1, y1, x2, y2) in transformed.get("bboxes", [])
                ]
                class_indices = valid_class_indices  # 与变换后的框保持顺序一致
            else:
                # 若所有 bbox 都无效,则仍需要对图像应用增强,但不参与检测分支。
                transformed = self.transform(image=image, bboxes=[], class_labels=[])
                image = transformed["image"]
                bboxes = []
                class_indices = []
        elif self.transform:
            # 没有 bbox 的样本,直接对图像做增强,而不调用 bbox 参数。
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed["image"]

        # 将复合标签展开成 (框, 类别) 对:一个框有多个类别时,框会被复制,
        # 使每个类别都作为独立的正样本参与检测训练。
        expanded_boxes = []
        expanded_labels = []
        for bbox, indices in zip(bboxes, class_indices):
            for idx in indices:
                expanded_boxes.append(bbox)
                expanded_labels.append(idx)

        # 将边界框列表转换成统一格式的 tensor,便于后续 DataLoader/Trainer 处理。
        if expanded_boxes:
            bboxes_tensor = torch.tensor(expanded_boxes, dtype=torch.float32)
        else:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        # 每个 (框, 类别) 对对应一个检测标签。
        if expanded_labels:
            lesion_labels_tensor = torch.tensor(expanded_labels, dtype=torch.long)
        else:
            lesion_labels_tensor = torch.zeros(0, dtype=torch.long)

        return {
            "image": image,                          # [C, H, W] 已归一化
            "disease_label": disease_label,           # int 0-2
            "position_label": position_label,         # int 0-3 (-1 = unknown/ignore)
            "lesion_multi_hot": torch.tensor(lesion_multi_hot, dtype=torch.float32),
            "bboxes": bboxes_tensor,                  # [N, 4]
            "lesion_labels": lesion_labels_tensor,    # [N]
            "image_id": Path(sample["image_path"]).stem,
            "orig_size": (orig_h, orig_w),
        }
