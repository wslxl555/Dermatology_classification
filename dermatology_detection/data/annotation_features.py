"""
从 .json 标注文件中提取皮损类型统计直方图（annotation statistics）。

这是两阶段流程第二阶段（疾病分类）的输入特征：
  - 对每张图片，读取其 LabelMe .json 标注；
  - 将每个标注框的原始标签解析为标准化的 15 类皮损类别；
  - 统计每类皮损出现的次数（count）与是否出现（multi-hot）；
  - 归一化 count 得到占比（proportion），作为统计特征。

训练阶段：直接来自 .json 标注。
推理阶段：由第一阶段检测模型输出的 box + annotation 计算得到。
"""

from pathlib import Path

import numpy as np

from data.label_mapping import LESION_CLASSES, LESION_CLASS_TO_IDX, parse_label
from utils.labelme_parser import parse_labelme


def load_lesion_histogram(
    json_path,
    num_classes: int | None = None,
) -> np.ndarray:
    """
    读取单个 .json，返回长度 num_classes 的计数直方图（float32）。

    多标签标注（如 "Plaque+Scale"）会对所有组成类别分别计数。
    无标注或 json_path 为空时返回全 0 向量。
    """
    num_classes = num_classes or len(LESION_CLASSES)
    counts = np.zeros(num_classes, dtype=np.float32)

    if not json_path:
        return counts

    parsed = parse_labelme(Path(json_path))
    if not parsed or not parsed.get("annotations"):
        return counts

    for ann in parsed["annotations"]:
        for cls_name in parse_label(ann["label"]):
            idx = LESION_CLASS_TO_IDX.get(cls_name)
            if idx is not None:
                counts[idx] += 1.0

    return counts


def histogram_to_features(counts: np.ndarray) -> dict:
    """
    把计数直方图转换为分类器输入特征。

    返回:
        "count":      [15] 每类皮损的标注框计数
        "proportion": [15] 归一化计数（除以总标注框数；无标注则为全 0）
        "multi_hot":  [15] 每类皮损是否出现（0/1）
    """
    counts = np.asarray(counts, dtype=np.float32)
    total = counts.sum()
    proportion = counts / total if total > 0 else np.zeros_like(counts)
    multi_hot = (counts > 0).astype(np.float32)
    return {
        "count": counts,
        "proportion": proportion,
        "multi_hot": multi_hot,
    }


def labels_to_histogram(
    lesion_labels,
    num_classes: int | None = None,
    scores=None,
) -> np.ndarray:
    """
    从检测模型输出的皮损类别索引（+ 可选置信度）构建计数直方图。

    用于推理阶段：把第一阶段检测到的 box + annotation 转成与训练阶段
    一致的统计特征。

    参数:
        lesion_labels: 长度为 N 的类别索引序列（int）。
        scores: 可选，长度为 N 的置信度序列；提供时按置信度加权计数。
    """
    num_classes = num_classes or len(LESION_CLASSES)
    counts = np.zeros(num_classes, dtype=np.float32)
    scores = [1.0] * len(lesion_labels) if scores is None else scores
    for lbl, sc in zip(lesion_labels, scores):
        idx = int(lbl)
        if 0 <= idx < num_classes:
            counts[idx] += float(sc)
    return counts
