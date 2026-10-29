"""
「分割式检测」头的公共底座。

多个检测头（TransUNet / TransCASCADE / Attention U-Net / Mamba / SAM-Adapter）
共享同一套外围管线，仅「瓶颈 + 解码器」不同：

  - 训练：bbox 目标渲染为逐像素 mask（弱监督），CE（背景降权）+ 前景 soft Dice。
  - 推理：argmax 得分割图，连通域分析把每个连通分量转成边界框。
  - get_lesion_features：瓶颈输出自适应池化 → [B, fpn_channels]，供疾病分类融合。

子类只需实现两个抽象方法：
  - _encode_bottleneck(bottleneck_feat) -> [B, d, h, w]  最深一级特征的瓶颈编码
  - _segment(feats, images) -> [B, C+1, H, W]            瓶颈 + 解码 → 分割 logits
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 解码器卷积块
# ---------------------------------------------------------------------------

class _DecoderBlock(nn.Module):
    """两个 3x3 卷积 + GroupNorm + ReLU 的解码块。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _render_masks(
    targets: list[dict],
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """
    将 bbox 目标渲染为逐像素分类 mask。

    返回:
        masks: [B, H, W] long —— 背景=0，框内部=label+1（1..C）。
    """
    B = len(targets)
    masks = torch.zeros(B, H, W, dtype=torch.long, device=device)

    for b, t in enumerate(targets):
        boxes = t.get("boxes", None)
        labels = t.get("labels", None)
        if boxes is None or len(boxes) == 0:
            continue

        for box, lbl in zip(boxes, labels):
            x1 = int(round(float(box[0])))
            y1 = int(round(float(box[1])))
            x2 = int(round(float(box[2])))
            y2 = int(round(float(box[3])))

            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue

            masks[b, y1:y2, x1:x2] = int(lbl) + 1

    return masks


def _soft_dice_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    前景类（1..C）的多类 soft Dice 损失。

    参数:
        logits: [B, C+1, H, W] 分割 logits。
        masks:  [B, H, W] long 目标 mask（背景=0）。
    """
    probs = logits.softmax(dim=1)  # [B, C+1, H, W]
    target = F.one_hot(masks, num_classes=num_classes + 1).permute(0, 3, 1, 2).float()

    dice = 0.0
    n = 0
    for c in range(1, num_classes + 1):
        p = probs[:, c]           # [B, H, W]
        t = target[:, c]          # [B, H, W]
        # 只对 batch 中实际出现的类别计算 Dice：否则"不存在类别"会被计满 1.0 的
        # 惩罚，强烈抑制稀有类的预测，加剧长尾分布下的漏检。
        if t.sum() == 0:
            continue
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice = dice + (1.0 - 2.0 * inter / denom.clamp(min=1e-6))
        n += 1

    if n == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return dice / n


# ---------------------------------------------------------------------------
# 公共底座
# ---------------------------------------------------------------------------

class _SegmentationHeadBase(nn.Module):
    """
    分割式检测头的公共底座，实现完整管线，仅瓶颈/解码器留给子类。

    参数:
        fpn_channels: 融合通道数（get_lesion_features 的输出维度）。
        num_classes: 皮损类别数（分割通道 = num_classes + 1，含背景）。
        d_model: 瓶颈维度；默认 = fpn_channels。
        min_area / score_thresh / max_detections: 连通域推理的过滤参数。
    """

    def __init__(
        self,
        fpn_channels: int,
        num_classes: int,
        d_model: int | None = None,
        min_area: int = 16,
        score_thresh: float = 0.05,
        max_detections: int = 100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.d_model = fpn_channels if d_model is None else d_model
        self.min_area = min_area
        self.score_thresh = score_thresh
        self.max_detections = max_detections

        # 背景降权的 CE 类别权重（背景=0.1，前景=1.0）
        self.register_buffer(
            "cls_weights",
            torch.tensor([0.1] + [1.0] * num_classes, dtype=torch.float32),
        )

        # 仅当 d_model != fpn_channels 时才需要投影，保证 get_lesion_features
        # 输出维度与 multi_task_model 期望的 fpn_channels 一致。
        self.lesion_proj = (
            nn.Linear(self.d_model, fpn_channels)
            if self.d_model != fpn_channels else None
        )

    # ------------------------------------------------------------------
    # 子类需实现的抽象方法
    # ------------------------------------------------------------------

    def _encode_bottleneck(self, bottleneck_feat: torch.Tensor) -> torch.Tensor:
        """最深一级特征 → [B, d, h, w] 瓶颈编码。"""
        raise NotImplementedError

    def _segment(self, feats: list[torch.Tensor], images: torch.Tensor) -> torch.Tensor:
        """瓶颈 + 解码 → [B, C+1, H, W] 分割 logits（已插值到输入分辨率）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 特征组织
    # ------------------------------------------------------------------

    @staticmethod
    def _ordered_features(features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """按 key 排序返回特征列表（stage1..stage4，stride 递增）。"""
        keys = sorted(features.keys())
        return [features[k] for k in keys]

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def forward(self, features, images, targets=None):
        feats = self._ordered_features(features)
        seg_logits = self._segment(feats, images)

        if targets is not None:
            return self._forward_train(seg_logits, images, targets)
        return self._forward_inference(seg_logits)

    def get_lesion_features(self, features, images) -> torch.Tensor:
        """瓶颈输出池化 → [B, fpn_channels]，供疾病分类头融合。"""
        feats = self._ordered_features(features)
        x = self._encode_bottleneck(feats[-1])
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # [B, d]
        if self.lesion_proj is not None:
            pooled = self.lesion_proj(pooled)
        return pooled

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def _forward_train(self, seg_logits, images, targets):
        device = images.device
        H, W = images.shape[-2:]
        masks = _render_masks(targets, H, W, device)

        ce = F.cross_entropy(seg_logits, masks, weight=self.cls_weights)
        dice = _soft_dice_loss(seg_logits, masks, self.num_classes)
        return {"losses": {"loss_det_seg": ce + dice}}

    # ------------------------------------------------------------------
    # 推理：argmax → 连通域 → 边界框
    # ------------------------------------------------------------------

    def _forward_inference(self, seg_logits):
        device = seg_logits.device
        B = seg_logits.shape[0]
        probs = seg_logits.softmax(dim=1)   # [B, C+1, H, W]
        pred = probs.argmax(dim=1)          # [B, H, W]

        all_boxes, all_scores, all_labels = [], [], []
        for b in range(B):
            pred_np = pred[b].cpu().numpy().astype(np.uint8)
            boxes_b, scores_b, labels_b = [], [], []

            for c in range(1, self.num_classes + 1):
                mask = (pred_np == c).astype(np.uint8)
                num, comp_map, stats, _ = cv2.connectedComponentsWithStats(
                    mask, connectivity=8
                )
                if num <= 1:
                    continue

                # 该类在该像素上的概率（背景之外），供组件置信度使用。
                class_prob = (probs[b, c] * (pred[b] == c).float()).cpu().numpy()

                for i in range(1, num):
                    x, y, w, h, area = stats[i]
                    if area < self.min_area:
                        continue
                    comp = comp_map[y:y + h, x:x + w] == i
                    score = float(class_prob[y:y + h, x:x + w][comp].mean())
                    if score < self.score_thresh:
                        continue
                    boxes_b.append([x, y, x + w, y + h])
                    scores_b.append(score)
                    labels_b.append(c - 1)

            boxes_t = (
                torch.tensor(boxes_b, dtype=torch.float32, device=device)
                if boxes_b else torch.zeros((0, 4), device=device)
            )
            scores_t = (
                torch.tensor(scores_b, dtype=torch.float32, device=device)
                if scores_b else torch.zeros(0, device=device)
            )
            labels_t = (
                torch.tensor(labels_b, dtype=torch.long, device=device)
                if labels_b else torch.zeros(0, dtype=torch.long, device=device)
            )

            # 与其它检测头一致：按分数截断到 top-k
            if scores_t.numel() > 0:
                top = torch.topk(scores_t, min(self.max_detections, scores_t.numel())).indices
                boxes_t, scores_t, labels_t = boxes_t[top], scores_t[top], labels_t[top]

            all_boxes.append(boxes_t)
            all_scores.append(scores_t)
            all_labels.append(labels_t)

        return {"boxes": all_boxes, "labels": all_labels, "scores": all_scores}
