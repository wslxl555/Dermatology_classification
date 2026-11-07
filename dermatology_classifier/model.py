"""
纯疾病分类模型：torchvision backbone + MLP 分类头。

结构：backbone（去掉分类层）→ 全局池化 → Flatten → MLP → 3 类 logits。
"""

import warnings

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base,
    swin_t, swin_s, resnet50, resnet101,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights,
    Swin_T_Weights, Swin_S_Weights, ResNet50_Weights, ResNet101_Weights,
)

# backbone 名 → (builder, weights 枚举, 特征维度, 分类层属性名)
BACKBONES = {
    "convnext_tiny":  (convnext_tiny,  ConvNeXt_Tiny_Weights,  768, "classifier"),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights, 768, "classifier"),
    "convnext_base":  (convnext_base,  ConvNeXt_Base_Weights,  1024, "classifier"),
    "swin_t":         (swin_t,         Swin_T_Weights,         768, "head"),
    "swin_s":         (swin_s,         Swin_S_Weights,         768, "head"),
    "resnet50":       (resnet50,       ResNet50_Weights,       2048, "fc"),
    "resnet101":      (resnet101,      ResNet101_Weights,      2048, "fc"),
}

BACKBONE_NAMES = list(BACKBONES.keys())


class ClassifierHead(nn.Module):
    """LayerNorm → Linear → GELU → Dropout → Linear → GELU → Dropout → Linear。"""

    def __init__(self, in_features: int, num_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ClassificationModel(nn.Module):
    """端到端纯疾病分类模型。"""

    def __init__(self, backbone_name: str = "convnext_tiny",
                 num_classes: int = 3, pretrained: bool = True,
                 offline: bool = False, hidden_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        builder, weights_enum, feat_dim, head_attr = BACKBONES[backbone_name]
        base = _build_pretrained(
            builder, weights_enum, backbone_name, pretrained, offline,
        )
        # 去掉原始分类层，保留特征输出
        setattr(base, head_attr, nn.Identity())

        self.backbone = base
        self.feat_dim = feat_dim
        self.head = ClassifierHead(feat_dim, num_classes, hidden_dim, dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        # 兼容 [B, C, 1, 1] 与 [B, C] 两种输出
        feats = torch.flatten(feats, 1)
        return self.head(feats)


def _is_network_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in (
        "name or service not known", "getaddrinfo", "urlopen error",
        "connection refused", "network is unreachable",
        "temporary failure in name resolution", "timed out",
    ))


def _build_pretrained(builder, weights_enum, name, pretrained, offline):
    """加载带预训练权重的 torchvision 模型，离线/网络失败时回退随机初始化。"""
    if not pretrained:
        return builder(weights=None)

    weights = weights_enum.IMAGENET1K_V1
    try:
        return builder(weights=weights)
    except Exception as e:
        if offline or _is_network_error(e):
            warnings.warn(
                f"无法联网加载 '{name}' 预训练权重，回退到随机初始化。"
                f"（可在有网环境先运行一次以缓存权重）"
            )
            return builder(weights=None)
        raise


def build_model(backbone_name: str = "convnext_tiny", num_classes: int = 3,
                pretrained: bool = True, offline: bool = False,
                hidden_dim: int = 512, dropout: float = 0.3) -> ClassificationModel:
    """构建纯疾病分类模型的工厂函数。"""
    return ClassificationModel(
        backbone_name=backbone_name, num_classes=num_classes,
        pretrained=pretrained, offline=offline,
        hidden_dim=hidden_dim, dropout=dropout,
    )
