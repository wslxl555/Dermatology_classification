"""
特征提取 backbone，支持可配置的网络架构。
支持 ConvNeXt、Swin/SwinV2、EfficientNet、ResNet（金字塔，可用于检测）
以及 ViT、DINOv2（单分辨率，仅用于疾病分类）。

离线 HPC 支持：如果网络不可用（例如没有互联网的计算节点），
构建函数会捕获 DNS/网络错误，并回退到缓存的权重或随机初始化，
而不是直接崩溃。
"""

import os
import warnings

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base,
    swin_t, swin_s, swin_b,
    swin_v2_t, swin_v2_s, swin_v2_b,
    resnet50, resnet101,
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l,
    vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights,
    Swin_T_Weights, Swin_S_Weights, Swin_B_Weights,
    Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights,
    ResNet50_Weights, ResNet101_Weights,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights,
    EfficientNet_B6_Weights, EfficientNet_B7_Weights,
    EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights,
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights,
    ViT_H_14_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor


def _is_network_error(error: Exception) -> bool:
    """检测来自 urllib / socket 层的 DNS 或连接失败。"""
    msg = str(error).lower()
    keywords = (
        "name or service not known", "getaddrinfo",
        "urlopen error", "connection refused",
        "network is unreachable", "temporary failure in name resolution",
        "no route to host", "timed out", "connection reset",
    )
    return any(kw in msg for kw in keywords)


def _safe_load(builder, weights, name: str, offline: bool):
    """
    加载带预训练权重的 torchvision 模型。

    当处于离线模式（显式指定或由网络错误检测到）时，
    该函数会回退到本地缓存；如果连缓存也没有，则返回一个
    随机初始化的模型并给出明确警告。
    """
    if weights is None:
        return builder(weights=None)

    if offline or os.environ.get("TORCH_HUB_OFFLINE") == "1":
        warnings.warn(
            f"Offline mode: loading '{name}' from local cache (if available) "
            f"or falling back to random init."
        )
        try:
            return builder(weights=weights)
        except Exception as e:
            if _is_network_error(e):
                warnings.warn(
                    f"Network unavailable and no cached weights for '{name}' — "
                    f"using random initialization. Run download_weights.py on "
                    f"the login node to cache pretrained weights."
                )
                return builder(weights=None)
            raise

    # 正常路径 —— 先尝试联网，遇到网络错误时回退
    try:
        return builder(weights=weights)
    except Exception as e:
        if _is_network_error(e):
            warnings.warn(
                f"Network unavailable for '{name}' — falling back to random "
                f"init. Run download_weights.py on the login node first.\n"
                f"  Error: {e}"
            )
            return builder(weights=None)
        raise


# ---------------------------------------------------------------------------
# 包装器：统一不同 backbone 的输出格式为 {"stage*": [B, C, H, W]}
# ---------------------------------------------------------------------------

class _SwinBackbone(nn.Module):
    """将 Swin/SwinV2 的 channels-last 特征 [B, H, W, C] 转为 [B, C, H, W]。"""

    def __init__(self, extractor: nn.Module):
        super().__init__()
        self.extractor = extractor

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.extractor(x)
        return {k: v.permute(0, 3, 1, 2).contiguous() for k, v in out.items()}


class _SequenceBackbone(nn.Module):
    """
    将 token 序列输出 [B, N, D] 的 backbone（ViT）池化为 [B, D, 1, 1]，
    使下游全局池化 + 分类头无需改动。
    """

    def __init__(self, extractor: nn.Module):
        super().__init__()
        self.extractor = extractor

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.extractor(x)
        # ViT 的 encoder 输出 [B, N+1, D]（含 CLS token）
        v = list(out.values())[0]
        v = v.mean(dim=1)  # [B, D]
        return {"stage4": v.unsqueeze(-1).unsqueeze(-1)}


class _TimmBackbone(nn.Module):
    """包装 timm 模型（DINOv2），输出池化后的 [B, D, 1, 1]。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.model.forward_features(x)  # [B, N, D]
        v = feats.mean(dim=1)  # [B, D]
        return {"stage4": v.unsqueeze(-1).unsqueeze(-1)}


def _infer_channels(extractor: nn.Module, keys: list[str]) -> list[int]:
    """对 channels-first 的 backbone，用一次 dry-forward 推断各 stage 通道数。"""
    with torch.no_grad():
        out = extractor(torch.zeros(1, 3, 224, 224))
    return [out[k].shape[1] for k in keys]


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

# 金字塔（多尺度）backbone —— 可用于检测；ViT/DINOv2 为单分辨率，仅分类。
BACKBONE_NAMES = [
    "convnext_tiny", "convnext_small", "convnext_base",
    "swin_t", "swin_s", "swin_b",
    "swin_v2_t", "swin_v2_s", "swin_v2_b",
    "resnet50", "resnet101",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
    "efficientnet_b6", "efficientnet_b7",
    "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
]

# 仅分类（单分辨率）backbone —— 前缀匹配
NON_PYRAMIDAL_PREFIXES = ("vit_", "dinov2_")


def backbone_supports_detection(name: str) -> bool:
    """返回该 backbone 是否输出多尺度金字塔特征（可用于检测头）。"""
    return not name.startswith(NON_PYRAMIDAL_PREFIXES)


def build_backbone(
    name: str = "convnext_tiny",
    pretrained: bool = True,
    offline: bool = False,
):
    """
    构建一个特征提取 backbone。

    返回:
        backbone: nn.Module 特征提取器，输出 {"stage*": [B, C, H, W]}。
        feature_channels: list[int] —— 每个 stage 的输出通道维度。
    """
    # ---- ConvNeXt（金字塔）----
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = _safe_load(convnext_tiny, weights, name, offline)
        backbone = create_feature_extractor(model, {
            "features.1": "stage1", "features.3": "stage2",
            "features.5": "stage3", "features.7": "stage4",
        })
        return backbone, [96, 192, 384, 768]

    elif name == "convnext_small":
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = _safe_load(convnext_small, weights, name, offline)
        backbone = create_feature_extractor(model, {
            "features.1": "stage1", "features.3": "stage2",
            "features.5": "stage3", "features.7": "stage4",
        })
        return backbone, [96, 192, 384, 768]

    elif name == "convnext_base":
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = _safe_load(convnext_base, weights, name, offline)
        backbone = create_feature_extractor(model, {
            "features.0": "stage1", "features.2": "stage2",
            "features.4": "stage3", "features.6": "stage4",
        })
        return backbone, [128, 256, 512, 1024]

    # ---- Swin / SwinV2（金字塔，channels-last → 需要 permute）----
    elif name in ("swin_t", "swin_s", "swin_b",
                  "swin_v2_t", "swin_v2_s", "swin_v2_b"):
        builder, weights_enum, channels = {
            "swin_t":    (swin_t, Swin_T_Weights, [96, 192, 384, 768]),
            "swin_s":    (swin_s, Swin_S_Weights, [96, 192, 384, 768]),
            "swin_b":    (swin_b, Swin_B_Weights, [128, 256, 512, 1024]),
            "swin_v2_t": (swin_v2_t, Swin_V2_T_Weights, [96, 192, 384, 768]),
            "swin_v2_s": (swin_v2_s, Swin_V2_S_Weights, [96, 192, 384, 768]),
            "swin_v2_b": (swin_v2_b, Swin_V2_B_Weights, [128, 256, 512, 1024]),
        }[name]
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        model = _safe_load(builder, weights, name, offline)
        extractor = create_feature_extractor(model, {
            "features.0": "stage1", "features.2": "stage2",
            "features.4": "stage3", "features.6": "stage4",
        })
        return _SwinBackbone(extractor), channels

    # ---- ResNet（金字塔）----
    elif name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = _safe_load(resnet50, weights, name, offline)
        backbone = create_feature_extractor(model, {
            "relu": "stage1", "layer1": "stage2", "layer2": "stage3",
            "layer3": "stage4", "layer4": "stage5",
        })
        return backbone, [64, 256, 512, 1024, 2048]

    elif name == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        model = _safe_load(resnet101, weights, name, offline)
        backbone = create_feature_extractor(model, {
            "relu": "stage1", "layer1": "stage2", "layer2": "stage3",
            "layer3": "stage4", "layer4": "stage5",
        })
        return backbone, [64, 256, 512, 1024, 2048]

    # ---- EfficientNet（金字塔）----
    elif name.startswith("efficientnet"):
        builders = {
            "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights),
            "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights),
            "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights),
            "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights),
            "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights),
            "efficientnet_b5": (efficientnet_b5, EfficientNet_B5_Weights),
            "efficientnet_b6": (efficientnet_b6, EfficientNet_B6_Weights),
            "efficientnet_b7": (efficientnet_b7, EfficientNet_B7_Weights),
            "efficientnet_v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights),
            "efficientnet_v2_m": (efficientnet_v2_m, EfficientNet_V2_M_Weights),
            "efficientnet_v2_l": (efficientnet_v2_l, EfficientNet_V2_L_Weights),
        }
        builder, weights_enum = builders[name]
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        model = _safe_load(builder, weights, name, offline)
        # 取 stride 4 / 8 / 16 / 32 的四个 stage；b* 与 v2* 的末块索引不同。
        if name.startswith("efficientnet_v2"):
            nodes = {"features.2": "stage1", "features.3": "stage2",
                     "features.5": "stage3", "features.7": "stage4"}
        else:
            nodes = {"features.2": "stage1", "features.3": "stage2",
                     "features.5": "stage3", "features.8": "stage4"}
        backbone = create_feature_extractor(model, nodes)
        channels = _infer_channels(backbone, ["stage1", "stage2", "stage3", "stage4"])
        return backbone, channels

    # ---- ViT（单分辨率，仅分类）----
    elif name.startswith("vit_"):
        builders = {
            "vit_b_16": (vit_b_16, ViT_B_16_Weights),
            "vit_b_32": (vit_b_32, ViT_B_32_Weights),
            "vit_l_16": (vit_l_16, ViT_L_16_Weights),
            "vit_l_32": (vit_l_32, ViT_L_32_Weights),
            "vit_h_14": (vit_h_14, ViT_H_14_Weights),
        }
        builder, weights_enum = builders[name]
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        model = _safe_load(builder, weights, name, offline)
        extractor = create_feature_extractor(model, {"encoder": "features"})
        hidden_dim = model.hidden_dim
        return _SequenceBackbone(extractor), [hidden_dim]

    # ---- DINOv2（单分辨率，仅分类，依赖 timm）----
    elif name.startswith("dinov2_"):
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                f"DINOv2 backbone '{name}' requires the 'timm' package. "
                f"Install with: pip install timm"
            ) from e
        timm_names = {
            "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
            "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
            "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
        }
        channels = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
        }
        try:
            model = timm.create_model(
                timm_names[name], pretrained=pretrained, num_classes=0,
            )
        except Exception as e:
            if _is_network_error(e) and not offline:
                warnings.warn(
                    f"Network unavailable for '{name}' — falling back to "
                    f"random init. Error: {e}"
                )
                model = timm.create_model(
                    timm_names[name], pretrained=False, num_classes=0,
                )
            else:
                raise
        model.eval()
        return _TimmBackbone(model), [channels[name]]

    else:
        raise ValueError(f"Unknown backbone: {name}")
