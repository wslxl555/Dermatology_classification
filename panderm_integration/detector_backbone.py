"""
把 PanDerm ViT 的 patch 特征接进检测头（权重就绪后使用）。

PanDerm 是 plain ViT（224px、patch16 → 14×14 网格）。本模块把它当作检测骨干：
把 patch token 还原成空间特征图，再用 torchvision 的 Faster R-CNN / FCOS 检测头
（即现有检测工程所用的同款头）做「皮损区域 + 类型」的检测。

重要限制（务必先读）：
  1. 原版 ViT 固定 224×224 输入、特征图仅 14×14，对小病灶（丘疹/斑疹）太粗。
     这只是一个「可跑的最小正确范式」，用于验证接法；真正要精度应改用
     mmdet 的 ViT-FPN（Derm1M 代码已带 mask_rcnn_vit_fpn / cascade_mask_rcnn_vit_fpn 配置），
     或把 ViT 的 pos_embed 插值到更高分辨率 / 用窗口注意力。
  2. 需要 timm（加载 PanDerm modeling_finetune）。权重就绪后运行。

用法:
    python -m panderm_integration.detector_backbone \
        --checkpoint PanDerm/pretrain_weight/panderm_bb_data6_checkpoint-499.pth \
        --variant base --head fasterrcnn --num_classes 16 --check
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn

from panderm_integration import model as pmodel


class PanDermBackbone(nn.Module):
    """把 PanDerm ViT 包装成 torchvision 检测骨干。

    forward 返回 OrderedDict[str, Tensor]，键 "0".."n" 对应多尺度特征图。
    单一尺度（默认）时只返回 "0"（patch 网格，stride=16）。
    """

    def __init__(self, model: nn.Module, embed_dim: int, scales=(1.0,)):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        # torchvision FasterRCNN/FCOS 依赖 backbone.out_channels
        self.out_channels = embed_dim
        self.scales = tuple(scales)
        # 为不同尺度准备 1×1 通道投影（保持 out_channels = embed_dim）
        self.projs = nn.ModuleDict({str(s): nn.Conv2d(embed_dim, embed_dim, 1)
                                    for s in self.scales})

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        feat = pmodel.extract_patch_features(self.model, x)  # (B, C, 14, 14)
        out = OrderedDict()
        for s in self.scales:
            f = feat
            if s < 1.0:                                   # 上采样（更细）
                f = nn.functional.interpolate(f, scale_factor=1 / s, mode="bilinear",
                                              align_corners=False)
            elif s > 1.0:                                 # 下采样（更粗）
                f = nn.functional.interpolate(f, scale_factor=1 / s, mode="bilinear",
                                              align_corners=False)
            out[str(s)] = self.projs[str(s)](f)
        return out


def build_detector(model: nn.Module, head: str = "fasterrcnn",
                   num_classes: int = 16, embed_dim: int = 768,
                   multi_scale: bool = False):
    """把 PanDerm ViT 接到 torchvision 检测头。

    参数:
        model: PanDerm ViT（pmodel.load_panderm 返回）。
        head: 'fasterrcnn' 或 'fcos'。
        num_classes: 含背景（15 前景 + 1 背景 = 16）。
        embed_dim: ViT 嵌入维（base=768, large=1024）。
        multi_scale: 是否输出多尺度特征图（stride 8/16/32）。
    """
    from torchvision.models.detection import FCOS, FasterRCNN

    if multi_scale:
        # scales: 0.5→stride32, 1.0→stride16, 2.0→stride8（s 是相对尺度）
        backbone = PanDermBackbone(model, embed_dim, scales=(0.5, 1.0, 2.0))
    else:
        backbone = PanDermBackbone(model, embed_dim, scales=(1.0,))

    # PanDerm 用 ImageNet 归一化；torchvision 检测器内部会按 image_mean/std 归一化，
    # 这里设成 ImageNet，使 ViT 收到的输入与其预训练一致。
    common = dict(
        min_size=224, max_size=224,
        image_mean=(0.485, 0.456, 0.406), image_std=(0.228, 0.224, 0.225),
    )
    if head == "fasterrcnn":
        return FasterRCNN(backbone, num_classes=num_classes, **common)
    if head == "fcos":
        return FCOS(backbone, num_classes=num_classes, **common)
    raise ValueError(f"未知 head: {head}（可选 fasterrcnn/fcos）")


def _check(checkpoint, variant, head, num_classes, multi_scale, device):
    """权重就绪后：构建骨干 + 检测头，跑一次前向，验证形状。"""
    model = pmodel.load_panderm(checkpoint, variant, device)
    embed_dim = pmodel.VARIANT_CFG[variant]["embed_dim"]
    detector = build_detector(model, head, num_classes, embed_dim, multi_scale)
    detector = detector.to(device).eval()

    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        if head == "fcos":
            loss_dict = detector(x, [dict(boxes=torch.zeros((0, 4), device=device),
                                          labels=torch.zeros((0,), dtype=torch.int64,
                                                             device=device))])
            print("FCOS 前向 OK；loss keys:", list(loss_dict.keys()))
        else:
            loss_dict = detector(x, [dict(boxes=torch.zeros((0, 4), device=device),
                                          labels=torch.zeros((0,), dtype=torch.int64,
                                                             device=device))])
            print("FasterRCNN 前向 OK；loss keys:", list(loss_dict.keys()))
    print("骨干输出尺度（feature map keys）:",
          list(PanDermBackbone(model, embed_dim,
                               scales=(0.5, 1.0, 2.0) if multi_scale else (1.0,))
               (x).keys()))


def main():
    ap = argparse.ArgumentParser(description="PanDerm ViT → 检测头")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--variant", choices=["base", "large"], default="base")
    ap.add_argument("--head", choices=["fasterrcnn", "fcos"], default="fasterrcnn")
    ap.add_argument("--num_classes", type=int, default=16, help="含背景（15 前景+1）")
    ap.add_argument("--multi_scale", action="store_true")
    ap.add_argument("--check", action="store_true", help="构建并跑一次前向验证")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint or pmodel.find_checkpoint(args.variant)
    if ckpt is None:
        raise SystemExit(
            "未找到 PanDerm 权重。请先下载（见 panderm_integration/README.md）并放到\n"
            "  PanDerm/pretrain_weight/，或用 --checkpoint 显式指定路径。"
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.check:
        _check(str(ckpt), args.variant, args.head, args.num_classes,
               args.multi_scale, device)
    else:
        # 仅构建模型并打印结构（不跑前向）
        model = pmodel.load_panderm(str(ckpt), args.variant, device)
        embed_dim = pmodel.VARIANT_CFG[args.variant]["embed_dim"]
        detector = build_detector(model, args.head, args.num_classes, embed_dim,
                                  args.multi_scale)
        print(detector)


if __name__ == "__main__":
    main()
