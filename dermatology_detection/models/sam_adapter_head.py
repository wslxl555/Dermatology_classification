"""
SAM-Adapter 风格（适配器式轻量版）的「分割式检测」头。

SAM-Adapter 的核心思想：不微调整个大预训练编码器，而是在冻结的编码器上插入
轻量 adapter 模块以适应新场景。本实现为**轻量适配器式**（非完整 SAM）：

  - 复用共享 ImageNet 预训练 convnext 编码器的输出特征（不加载 SAM ViT 权重、
    不引入 segment-anything 依赖）。
  - 对每个 stage 特征（瓶颈 + 3 个跳连）各接一个残差 MLP adapter
    （AdaptFormer/LoRA 风格，下投影 r=4），做轻量适配。
  - 解码器为标准 U-Net（upsample ⊕ skip concat），分割头 1x1 conv。

说明：由于检测头只接收骨干的输出特征（而非骨干各层本身），adapter 作用在
特征层上，而非字面插入 convnext 内部层——保持 head-only、不改 backbone。

与 TransUNet 系共用同一套「分割式检测」管线（见 models/segmentation_base.py）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segmentation_base import _SegmentationHeadBase, _DecoderBlock


class _AdapterBlock(nn.Module):
    """AdaptFormer/LoRA 风格的残差瓶颈 MLP adapter（1x1 卷积实现）。"""

    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        hidden = max(channels // r, 1)
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)


class SAMAdapterHead(_SegmentationHeadBase):
    """SAM-Adapter 风格（适配器式）分割式检测头。"""

    def __init__(
        self,
        in_channels_list: list[int],
        fpn_channels: int,
        num_classes: int,
        d_model: int | None = None,
        adapter_r: int = 4,
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
        self.bottleneck_adapter = _AdapterBlock(d, adapter_r)
        self.skip_projs = nn.ModuleList(
            [nn.Conv2d(c, d, 1) for c in skip_channels]
        )
        self.skip_adapters = nn.ModuleList(
            [_AdapterBlock(d, adapter_r) for _ in range(3)]
        )
        self.dec_blocks = nn.ModuleList(
            [_DecoderBlock(2 * d, d) for _ in range(3)]
        )
        self.seg_head = nn.Conv2d(d, num_classes + 1, 1)

    def _encode_bottleneck(self, bottleneck_feat: torch.Tensor) -> torch.Tensor:
        """1x1 投影 + adapter → [B, d, h, w]。"""
        return self.bottleneck_adapter(self.bottleneck_proj(bottleneck_feat))

    def _segment(self, feats: list[torch.Tensor], images: torch.Tensor) -> torch.Tensor:
        bottleneck_feat = feats[-1]                 # stride 32
        skip_feats = feats[:-1][::-1]               # stride 16 / 8 / 4

        enc_feats = [self._encode_bottleneck(bottleneck_feat)]
        for i, f in enumerate(skip_feats):
            enc_feats.append(self.skip_adapters[i](self.skip_projs[i](f)))

        x = enc_feats[0]                            # stride 32
        for i in range(3):
            up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.dec_blocks[i](torch.cat([up, enc_feats[i + 1]], dim=1))

        logits = self.seg_head(x)  # [B, C+1, H/4, W/4]
        H, W = images.shape[-2:]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
