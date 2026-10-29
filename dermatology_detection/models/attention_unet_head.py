"""
Attention U-Net 风格的「分割式检测」头。

Attention U-Net（Oktay et al., 2018）在 U-Net 解码器的跳连上加入注意力门
（Attention Gate, AG）：用较粗层的门控信号对同层跳连特征做空间加权，抑制背景
区域的响应，使解码器聚焦于病灶区域。

架构：
  - 编码器：复用共享骨干（外部传入 stage1..stage4 金字塔特征）。
  - 瓶颈：纯卷积（1x1 投影 + 两个 3x3 卷积），无 Transformer。
  - 解码器：3 级上采样，每级先用注意力门对跳连加权（gating=上采样后的粗层特征），
    再 concat → 解码卷积块。
  - 分割头：1x1 conv → num_classes + 1 通道（含背景）。

与 TransUNet 系共用同一套「分割式检测」管线（见 models/segmentation_base.py）：
box→mask 弱监督训练、连通域推理、get_lesion_features 池化瓶颈。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segmentation_base import _SegmentationHeadBase, _DecoderBlock


class _AttentionGate(nn.Module):
    """
    注意力门：用门控信号 g 对跳连特征 x 做空间加权。

    q_att = ψ(ReLU(Wg·g + Wx·x))   （1x1 卷积，F_int = d//2）
    α     = sigmoid(q_att)
    输出   = x * α
    """

    def __init__(self, channels: int, inter_channels: int | None = None):
        super().__init__()
        inter_channels = inter_channels or channels // 2
        self.W_g = nn.Conv2d(channels, inter_channels, 1, bias=False)
        self.W_x = nn.Conv2d(channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g：门控信号（较粗层上采样后）；x：同层跳连特征。
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        q = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        alpha = torch.sigmoid(q)          # [B, 1, H, W]
        return x * alpha


class AttentionUNetHead(_SegmentationHeadBase):
    """Attention U-Net 风格分割式检测头：卷积瓶颈 + 注意力门控跳连解码器。"""

    def __init__(
        self,
        in_channels_list: list[int],
        fpn_channels: int,
        num_classes: int,
        d_model: int | None = None,
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
        d = self.d_model

        skip_channels = in_channels_list[:-1][::-1]
        bottleneck_channels = in_channels_list[-1]

        self.bottleneck_proj = nn.Conv2d(bottleneck_channels, d, 1)
        self.bottleneck_conv = _DecoderBlock(d, d)
        self.skip_projs = nn.ModuleList(
            [nn.Conv2d(c, d, 1) for c in skip_channels]
        )
        self.attention_gates = nn.ModuleList(
            [_AttentionGate(d) for _ in range(3)]
        )
        self.dec_blocks = nn.ModuleList(
            [_DecoderBlock(2 * d, d) for _ in range(3)]
        )
        self.seg_head = nn.Conv2d(d, num_classes + 1, 1)

    def _encode_bottleneck(self, bottleneck_feat: torch.Tensor) -> torch.Tensor:
        """1x1 投影 + 卷积块 → [B, d, h, w]。"""
        return self.bottleneck_conv(self.bottleneck_proj(bottleneck_feat))

    def _segment(self, feats: list[torch.Tensor], images: torch.Tensor) -> torch.Tensor:
        bottleneck_feat = feats[-1]                 # stride 32
        skip_feats = feats[:-1][::-1]               # stride 16 / 8 / 4

        enc_feats = [self._encode_bottleneck(bottleneck_feat)]
        for i, f in enumerate(skip_feats):
            enc_feats.append(self.skip_projs[i](f))

        x = enc_feats[0]                            # stride 32
        for i in range(3):
            up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            attended = self.attention_gates[i](g=up, x=enc_feats[i + 1])
            x = self.dec_blocks[i](torch.cat([up, attended], dim=1))

        logits = self.seg_head(x)  # [B, C+1, H/4, W/4]
        H, W = images.shape[-2:]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
