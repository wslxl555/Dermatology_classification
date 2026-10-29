"""
目标检测评估指标：COCO/VOC 风格的平均精度（AP / mAP）。

对每个类别，在指定 IoU 阈值下计算 AP（101 点插值），
再对所有类别取平均得到 mAP。可扩展到多个 IoU 阈值。
"""

from __future__ import annotations

import numpy as np
import torch


def box_iou_np(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """计算 [N,4] 与 [M,4] 的成对 IoU（x1,y1,x2,y2）。"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.where(union > 0, union, 1e-6)


def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """由 PR 曲线计算 AP（101 点插值）。"""
    if len(precisions) == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        p = precisions[recalls >= t]
        ap += float(p.max()) if len(p) > 0 else 0.0
    return float(ap / 101.0)


def compute_map_per_class(
    predictions: list[dict],
    ground_truths: list[dict],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """
    在单个 IoU 阈值下计算每类 AP 与 mAP。

    参数:
        predictions: 每图一个 dict，含 "boxes" [N,4]、"scores" [N]、"labels" [N]。
        ground_truths: 每图一个 dict，含 "boxes" [M,4]、"labels" [M]。
        num_classes: 皮损类别数（标签索引 0..num_classes-1）。
        iou_threshold: 判定真阳性的 IoU 阈值。

    返回:
        {"mAP": float, "AP_<cls>": float, ...}
    """
    # 收集每类的 GT 框与预测
    gt_per_class = [[] for _ in range(num_classes)]
    for g in ground_truths:
        boxes = np.asarray(g["boxes"], dtype=np.float32)
        labels = np.asarray(g["labels"], dtype=np.int64)
        for c in range(num_classes):
            sel = boxes[labels == c] if len(boxes) > 0 else np.zeros((0, 4), np.float32)
            gt_per_class[c].append(sel)

    # 每类所有预测（含所属图像索引）
    pred_per_class = [[] for _ in range(num_classes)]
    for img_idx, p in enumerate(predictions):
        boxes = np.asarray(p["boxes"], dtype=np.float32)
        scores = np.asarray(p["scores"], dtype=np.float32)
        labels = np.asarray(p["labels"], dtype=np.int64)
        if len(boxes) == 0:
            continue
        for c in range(num_classes):
            m = labels == c
            for i in np.where(m)[0]:
                pred_per_class[c].append((img_idx, scores[i], boxes[i]))

    aps = {}
    for c in range(num_classes):
        preds = sorted(pred_per_class[c], key=lambda x: -x[1])
        # 每张图该类 GT 已匹配状态
        gt_matched = [np.zeros(len(gt), dtype=bool) for gt in gt_per_class[c]]
        n_gt = sum(len(g) for g in gt_per_class[c])

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        for j, (img_idx, _score, box) in enumerate(preds):
            gt_boxes = gt_per_class[c][img_idx]
            if len(gt_boxes) == 0:
                fp[j] = 1.0
                continue
            ious = box_iou_np(box[None, :], gt_boxes)[0]
            best = int(ious.argmax())
            if ious[best] >= iou_threshold and not gt_matched[img_idx][best]:
                tp[j] = 1.0
                gt_matched[img_idx][best] = True
            else:
                fp[j] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
        aps[f"AP_cls{c}"] = compute_ap(precisions, recalls)

    aps["mAP"] = float(np.mean([v for k, v in aps.items() if k.startswith("AP_")]))
    return aps


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    num_classes: int,
    iou_thresholds: tuple[float, ...] = (0.5, 0.75),
) -> dict[str, float]:
    """在多个 IoU 阈值下计算 mAP，返回 mAP@0.5 / mAP@0.75 / mAP@[.5,.75]。"""
    per_thresh = {}
    for t in iou_thresholds:
        per_thresh[t] = compute_map_per_class(
            predictions, ground_truths, num_classes, iou_threshold=t)

    out = {}
    for t in iou_thresholds:
        out[f"mAP@{t}"] = per_thresh[t]["mAP"]
        # 每个阈值下的每类 AP 也输出（便于诊断）
        for k, v in per_thresh[t].items():
            if k != "mAP":
                out[f"{k}@{t}"] = v
    out["mAP@[.5,.75]"] = float(np.mean([out[f"mAP@{t}"] for t in iou_thresholds]))
    return out
