"""
TransUNet / TransCASCADE 风格的「分割式检测」头。

与现有框回归类头（FCOS / Faster R-CNN / YOLOv8 / RT-DETR）互补：
本头输出逐像素的分割图（15 类 + 背景），推理时用连通域分析把每个
连通分量转成边界框（detection-by-segmentation），对粘连/不规则病灶更鲁棒。

架构：
  - 编码器：复用共享骨干（外部传入 stage1..stage4 金字塔特征），
    跳连取前 3 级，瓶颈取最深一级。
  - Transformer 瓶颈：最深特征 1x1 投影 + 2D 正弦位置编码 → 展平成
    token 序列 → nn.TransformerEncoder（轻量，默认 4 层）→ reshape 回 2D。
  - 解码器：3 级上采样恢复到 stride 4，再插值到输入分辨率。
    - TransUNet：每级 `upsample(prev) ⊕ 同层 skip`（concat）。
    - TransCASCADE：每级用 SKCA 级联 skip 聚合（聚合瓶颈 + 该级及之前
      的多尺度 skip，逐元素相加）替代单一 concat。
  - 分割头：末端 1x1 conv → num_classes + 1 通道（含背景）。

监督说明：数据集仅提供 bbox（无像素级 mask），因此训练 mask 由
「框→掩码」渲染（填充框内部作为前景）。这是弱监督分割。

统一接口（与其它头一致）：
  - forward(features, images, targets) → {"losses"} 或 {"boxes","labels","scores"}
  - get_lesion_features(features, images) → [B, fpn_channels]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detection_rtdetr import _pos_encoding_2d
from models.segmentation_base import (
    _SegmentationHeadBase,
    _DecoderBlock,
    _render_masks,
    _soft_dice_loss,
)

# 保留导出，便于外部按旧路径 import
__all__ = [
    "_TransformerUNetBase", "TransUNetHead", "TransCASCADEHead",
    "_DecoderBlock", "_render_masks", "_soft_dice_loss",
]


# ---------------------------------------------------------------------------
# 共享底座
# ---------------------------------------------------------------------------

class _TransformerUNetBase(_SegmentationHeadBase):
    """
    TransUNet 系分割式检测头的共享底座。

    参数:
        in_channels_list: 骨干各级通道数（stage1..stage4，stride 4/8/16/32）。
        fpn_channels: 融合通道数（也用于 get_lesion_features 的输出维度）。
        num_classes: 皮损类别数（分割通道 = num_classes + 1，含背景）。
        d_model: Transformer 维度；默认 = fpn_channels。
        num_layers / nhead / dim_ff: Transformer 瓶颈配置。
        cascade_aggregation: True 时使用 TransCASCADE 的 SKCA 级联 skip 聚合。
    """

    def __init__(
        self,
        in_channels_list: list[int],
        fpn_channels: int,
        num_classes: int,
        d_model: int | None = None,
        num_layers: int = 4,
        nhead: int = 8,
        dim_ff: int = 1024,
        cascade_aggregation: bool = False,
        min_area: int = 16,
        score_thresh: float = 0.05,
        max_detections: int = 100,
    ):
        assert len(in_channels_list) >= 4, "need at least 4 pyramid stages"
        super().__init__(
            fpn_channels, num_classes, d_model=d_model,
            min_area=min_area, score_thresh=score_thresh,
            max_detections=max_detections,
        )
        self.cascade_aggregation = cascade_aggregation
        d = self.d_model

        # 跳连取前 3 级（stride 16/8/4），瓶颈取最深一级（stride 32）。
        # 解码器由粗到细消费跳连（stride 16 → 8 → 4），故通道顺序也要反转。
        skip_channels = in_channels_list[:-1][::-1]
        bottleneck_channels = in_channels_list[-1]

        self.bottleneck_proj = nn.Conv2d(bottleneck_channels, d, 1)
        self.skip_projs = nn.ModuleList(
            [nn.Conv2d(c, d, 1) for c in skip_channels]
        )

        enc_layer = nn.TransformerEncoderLayer(
            d, nhead, dim_feedforward=dim_ff, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)

        # 3 级解码器：stride 32 → 16 → 8 → 4
        self.dec_blocks = nn.ModuleList(
            [_DecoderBlock(2 * d, d) for _ in range(3)]
        )

        # 分割头：num_classes 前景类 + 背景
        self.seg_head = nn.Conv2d(d, num_classes + 1, 1)

    # ------------------------------------------------------------------
    # 瓶颈 + 解码 → 分割 logits
    # ------------------------------------------------------------------

    def _aggregate_skips(self, enc_feats: list[torch.Tensor], size) -> torch.Tensor:
        """把多尺度编码特征上采样到同一尺度并逐元素相加（SKCA 聚合）。"""
        agg = None
        for e in enc_feats:
            up = F.interpolate(e, size=size, mode="bilinear", align_corners=False)
            agg = up if agg is None else agg + up
        return agg

    def _encode_bottleneck(self, bottleneck_feat: torch.Tensor) -> torch.Tensor:
        """1x1 投影 + 位置编码 + Transformer → [B, d, h, w]。"""
        x = self.bottleneck_proj(bottleneck_feat)  # [B, d, h, w]
        h, w = x.shape[-2:]
        pe = _pos_encoding_2d(self.d_model, h, w, x.device)
        x = x + pe.unsqueeze(0)

        B, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, h*w, d]
        x = self.transformer(x)                   # [B, h*w, d]
        x = x.transpose(1, 2).reshape(B, d, h, w)  # [B, d, h, w]
        return x

    def _segment(self, feats: list[torch.Tensor], images: torch.Tensor) -> torch.Tensor:
        bottleneck_feat = feats[-1]                 # stride 32
        skip_feats = feats[:-1][::-1]               # stride 16 / 8 / 4

        # 编码特征（d 通道）：瓶颈 + 各级跳连（由粗到细）
        enc_feats = [self._encode_bottleneck(bottleneck_feat)]
        for i, f in enumerate(skip_feats):
            enc_feats.append(self.skip_projs[i](f))

        # 解码器：3 级上采样（32→16→8→4）
        x = enc_feats[0]
        for i in range(3):
            up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            if self.cascade_aggregation:
                skip = self._aggregate_skips(enc_feats[: i + 2], up.shape[-2:])
            else:
                skip = enc_feats[i + 1]
            x = self.dec_blocks[i](torch.cat([up, skip], dim=1))

        logits = self.seg_head(x)  # [B, C+1, H/4, W/4]
        H, W = images.shape[-2:]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# 两个具体头
# ---------------------------------------------------------------------------

class TransUNetHead(_TransformerUNetBase):
    """经典 TransUNet 风格：解码器使用同层 skip concat。"""

    def __init__(self, in_channels_list, fpn_channels, num_classes, **kwargs):
        super().__init__(
            in_channels_list, fpn_channels, num_classes,
            cascade_aggregation=False, **kwargs,
        )


class TransCASCADEHead(_TransformerUNetBase):
    """TransCASCADE 风格：解码器使用 SKCA 级联 skip 聚合。"""

    def __init__(self, in_channels_list, fpn_channels, num_classes, **kwargs):
        super().__init__(
            in_channels_list, fpn_channels, num_classes,
            cascade_aggregation=True, **kwargs,
        )
