"""
Mamba（选择性状态空间模型, S6）风格的「分割式检测」头。

用 Mamba 块替代 TransUNet 的 Transformer 瓶颈：最深特征展平成 token 序列，
经若干 Mamba 块做序列建模，再 reshape 回 2D 进入 U-Net 解码器。解码器与
TransUNet 相同（upsample ⊕ skip concat）。

Mamba 块为**纯 PyTorch 实现**（顺序扫描），不依赖 mamba-ssm / causal-conv1d
CUDA kernel，可在 CPU 上训练/推理。瓶颈序列在 stride32 下仅 8x8=64 token，
顺序扫描开销可忽略。为保持轻量，省略了原始 Mamba 块中的深度因果卷积预处理
（仅保留 LayerNorm + 门控 + S6 选择性扫描 + 输出投影 + 残差）。

与 TransUNet 系共用同一套「分割式检测」管线（见 models/segmentation_base.py）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detection_rtdetr import _pos_encoding_2d
from models.segmentation_base import _SegmentationHeadBase, _DecoderBlock


class MambaBlock(nn.Module):
    """
    单个 Mamba 块（LayerNorm → 门控 + S6 选择性扫描 → 输出投影 → 残差）。

    参数:
        d_model: 特征维度。
        d_state: SSM 状态维度 N（默认 16）。
        expand: 内部维度放大倍数（默认 2）。
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        dt_rank = max(d_model // 16, 1)

        self.norm = nn.LayerNorm(d_model)
        # 门控分支 z 与输入分支 x 合并在一个投影里
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # 选择性参数：Δ 的低秩输入 + B + C
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # A：可学习的对数空间对角 SSM 参数 [d_inner, d_state]
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # D：跳跃连接 [d_inner]
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, u, delta, A, B, C, D):
        """S6 选择性扫描（顺序实现，精确）。"""
        B_, L, d = u.shape
        n = A.shape[-1]
        deltaA = torch.exp(delta[..., None] * A)      # [B, L, d, n]
        deltaB = delta[..., None] * B[..., None, :]   # [B, L, d, n]

        h = u.new_zeros(B_, d, n)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB[:, t] * u[:, t, :, None]
            y = (h * C[:, t, None, :]).sum(-1) + D * u[:, t]  # [B, d]
            ys.append(y)
        return torch.stack(ys, dim=1)  # [B, L, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                              # [B, L, 2*d_inner]
        u, z = xz.split([self.d_inner, self.d_inner], dim=-1)

        A = -torch.exp(self.A_log)                        # [d_inner, d_state]
        deltaBC = self.x_proj(u)                          # [B, L, rank+2n]
        delta, B, C = deltaBC.split(
            [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))           # [B, L, d_inner]

        y = self._selective_scan(u, delta, A, B, C, self.D)
        y = y * F.silu(z)
        return self.out_proj(y) + residual


class MambaHead(_SegmentationHeadBase):
    """Mamba 风格分割式检测头：Mamba 瓶颈 + U-Net 解码器（skip concat）。"""

    def __init__(
        self,
        in_channels_list: list[int],
        fpn_channels: int,
        num_classes: int,
        d_model: int | None = None,
        num_layers: int = 4,
        d_state: int = 16,
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
        self.skip_projs = nn.ModuleList(
            [nn.Conv2d(c, d, 1) for c in skip_channels]
        )
        self.mamba = nn.Sequential(
            *[MambaBlock(d, d_state) for _ in range(num_layers)]
        )
        self.dec_blocks = nn.ModuleList(
            [_DecoderBlock(2 * d, d) for _ in range(3)]
        )
        self.seg_head = nn.Conv2d(d, num_classes + 1, 1)

    def _encode_bottleneck(self, bottleneck_feat: torch.Tensor) -> torch.Tensor:
        """1x1 投影 + 位置编码 + Mamba 块 → [B, d, h, w]。"""
        x = self.bottleneck_proj(bottleneck_feat)  # [B, d, h, w]
        h, w = x.shape[-2:]
        pe = _pos_encoding_2d(self.d_model, h, w, x.device)
        x = x + pe.unsqueeze(0)

        B, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, h*w, d]
        x = self.mamba(x)                         # [B, h*w, d]
        x = x.transpose(1, 2).reshape(B, d, h, w)  # [B, d, h, w]
        return x

    def _segment(self, feats: list[torch.Tensor], images: torch.Tensor) -> torch.Tensor:
        bottleneck_feat = feats[-1]                 # stride 32
        skip_feats = feats[:-1][::-1]               # stride 16 / 8 / 4

        enc_feats = [self._encode_bottleneck(bottleneck_feat)]
        for i, f in enumerate(skip_feats):
            enc_feats.append(self.skip_projs[i](f))

        x = enc_feats[0]                            # stride 32
        for i in range(3):
            up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.dec_blocks[i](torch.cat([up, enc_feats[i + 1]], dim=1))

        logits = self.seg_head(x)  # [B, C+1, H/4, W/4]
        H, W = images.shape[-2:]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
