"""
PanDerm 权重就绪后：加载 PanDerm ViT 并提取 CLS / patch 特征。

PanDerm 是一个在 200 万张皮肤病图像上自监督预训练的 ViT 基础模型。
本模块封装了权重加载与特征提取，供 linear probe（linear_probe.py）与
检测骨干接入（detector_backbone.py）复用。

注意：
  * 加载模型需要 `timm`（PanDerm 的 modeling_finetune.py 依赖 timm.layers）。
  * 权重文件尚未下载（见 PANDERM_DOWNLOAD）。本模块对「权重缺失」做友好报错。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# PanDerm 模型定义所在目录（modeling_finetune.py 自包含，仅依赖 timm）
PANDERM_MODELS_DIR = (
    Path(__file__).resolve().parent.parent
    / "PanDerm" / "classification" / "models"
)

# PanDerm 预训练权重下载地址（官方 README 提供）
PANDERM_DOWNLOAD = {
    "base": (
        "PanDerm_Base (ViT-B/16)  文件名 panderm_bb_data6_checkpoint-499.pth\n"
        "  Google Drive: https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view"
    ),
    "large": (
        "PanDerm (ViT-L/16)        文件名 panderm_ll_data6_checkpoint-499.pth\n"
        "  Google Drive: https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view"
    ),
}

# ViT 架构参数
VARIANT_CFG = {
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
}

# 常见权重存放位置（自动搜索）
DEFAULT_CHECKPOINT_DIRS = [
    Path(__file__).resolve().parent.parent / "PanDerm" / "pretrain_weight",
    Path(__file__).resolve().parent.parent / "PanDerm",
    Path(__file__).resolve().parent.parent,
]

_MODELING_MODULE = None


def _load_modeling_module():
    """惰性导入 PanDerm 的 modeling_finetune（需要 timm）。"""
    global _MODELING_MODULE
    if _MODELING_MODULE is not None:
        return _MODELING_MODULE
    if not PANDERM_MODELS_DIR.exists():
        raise FileNotFoundError(
            f"未找到 PanDerm 模型定义目录: {PANDERM_MODELS_DIR}"
        )
    if str(PANDERM_MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(PANDERM_MODELS_DIR))
    try:
        import modeling_finetune as mf
    except ImportError as e:
        raise ImportError(
            "加载 PanDerm 需要 `timm`（modeling_finetune.py 依赖 timm.layers）。\n"
            f"  请先安装: pip install timm\n  原始错误: {e}"
        ) from e
    _MODELING_MODULE = mf
    return mf


def build_panderm(variant: str = "base") -> nn.Module:
    """构建 PanDerm ViT（随机初始化，不含预训练权重）。"""
    if variant not in VARIANT_CFG:
        raise ValueError(f"未知 variant: {variant}（可选 base/large）")
    mf = _load_modeling_module()
    if variant == "base":
        return mf.panderm_base_patch16_224(num_classes=0)
    return mf.panderm_large_patch16_224(num_classes=0)


def _extract_state_dict(ckpt) -> dict:
    """从 checkpoint 中提取纯 state_dict（兼容 model/state_dict/裸 dict）。"""
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("model", "state_dict")):
        ckpt = ckpt.get("model", ckpt.get("state_dict"))
    if not isinstance(ckpt, dict):
        raise ValueError(f"无法解析 checkpoint，类型 {type(ckpt)}")
    return ckpt


def _strip_prefix(state_dict: dict) -> dict:
    """去掉 'encoder.' / 'module.' 前缀（PanDerm Large 用 encoder.，DDP 用 module.）。"""
    out = {}
    for k, v in state_dict.items():
        kk = k
        for pref in ("encoder.", "module.", "backbone."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
                break
        out[kk] = v
    return out


def load_panderm(checkpoint: str, variant: str = "base",
                 device: str = "cpu") -> nn.Module:
    """加载 PanDerm ViT 预训练权重。

    参数:
        checkpoint: .pth 权重路径。
        variant: 'base'（ViT-B/16）或 'large'（ViT-L/16）。
        device: 'cpu' / 'cuda'。
    """
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"PanDerm 权重不存在: {checkpoint}\n"
            f"请先下载对应权重（当前网络受限，需在有网/已下载环境运行）：\n"
            f"  {PANDERM_DOWNLOAD[variant]}"
        )
    model = build_panderm(variant)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = _strip_prefix(_extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # 只对「权重里缺了 ViT 关键部分」报错，head 缺失是正常的（num_classes=0 时 head=Identity）
    critical_missing = [k for k in missing if "head" not in k]
    if critical_missing:
        raise RuntimeError(
            f"PanDerm 权重加载失败，缺失关键参数（前 10 个）:\n"
            f"  {critical_missing[:10]}\n"
            f"  请确认 {checkpoint} 与 variant='{variant}' 匹配。"
        )
    if unexpected:
        print(f"[warn] 权重含 {len(unexpected)} 个未使用键（可能是 head/分类头，正常）。")
    model.to(device).eval()
    return model


def find_checkpoint(variant: str = "base") -> Path | None:
    """在常见目录搜索 PanDerm 权重文件，找不到返回 None。"""
    names = {
        "base": ["panderm_bb_data6_checkpoint-499.pth"],
        "large": ["panderm_ll_data6_checkpoint-499.pth"],
    }[variant]
    for d in DEFAULT_CHECKPOINT_DIRS:
        if not d.exists():
            continue
        for n in names:
            p = d / n
            if p.exists():
                return p
        # 也尝试该目录下任意匹配的 .pth
        for p in sorted(d.glob("*.pth")):
            if variant in p.name.lower() or "panderm" in p.name.lower():
                return p
    return None


# =============================================================================
# 特征提取（CLS token / patch tokens）
# =============================================================================
@torch.no_grad()
def forward_tokens(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """前向得到完整 token 序列 (B, 1+N, C)：第 0 个是 CLS，其余为 patch。"""
    x = model.patch_embed(x)
    B, N, C = x.shape
    cls = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = x + model.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
    x = model.pos_drop(x)
    rel_pos_bias = model.rel_pos_bias() if model.rel_pos_bias is not None else None
    for blk in model.blocks:
        x = blk(x, rel_pos_bias=rel_pos_bias)
    x = model.norm(x)
    return x


@torch.no_grad()
def extract_cls_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """提取 CLS token 特征 (B, embed_dim)，用于 linear probe。"""
    tokens = forward_tokens(model, x)
    return tokens[:, 0]


@torch.no_grad()
def extract_patch_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """提取 patch token 特征并 reshape 为空间网格 (B, C, H, W)。

    用于检测骨干：把 ViT 的 patch token 当作特征图。H=W=img_size//16（224px → 14）。
    """
    tokens = forward_tokens(model, x)          # (B, 1+N, C)
    patches = tokens[:, 1:, :]                  # (B, N, C)
    B, N, C = patches.shape
    H = W = int(round(math.sqrt(N)))
    assert H * W == N, f"patch 数量 {N} 不是完全平方，无法还原为网格"
    return patches.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
