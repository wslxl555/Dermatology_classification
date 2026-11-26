"""
特征提取工具：图像变换、特征提取与缓存、以及一个「假编码器」用于无权重时
验证整条 linear-probe 流水线（无需 timm / PanDerm 权重）。

真实用法：权重就绪后，linear_probe.py 会自动走 model.load_panderm；
本模块的 DummyEncoder 只用于 `--dummy` 冒烟测试。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def build_eval_transform(norm: str = "imagenet", img_size: int = 224):
    """PanDerm 官方 eval 变换：Resize(256) → CenterCrop(224) → ToTensor → Normalize。"""
    if norm == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.228, 0.224, 0.225)
    elif norm == "uniform":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif norm == "clip":
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    else:
        raise ValueError(f"未知 norm: {norm}")
    # 先 resize 到 256 再中心裁 224，与 PanDerm linear_eval 一致
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class DummyEncoder(nn.Module):
    """用于无权重时验证流水线的假编码器（固定 3×3 卷积 → 全局池化 → 线性投影）。

    不需要 timm，不依赖 PanDerm。输出 dim 维特征，供 LogisticRegression 训练。
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.proj = nn.Linear(32, dim)
        # 固定初始化，保证可复现
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = F.relu(self.conv(x))
        h = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
        return self.proj(h)


def extract_features_from_paths(model, transform, paths, device="cpu",
                                batch_size=64, cache_path=None):
    """对图像路径列表提取 CLS 特征 (N, D)，可选缓存到 .npz。

    返回 (features[N, D], paths)。paths 与 features 一一对应。
    """
    if cache_path is not None and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["features"].astype(np.float32), list(data["paths"])

    model = model.to(device).eval()
    feats: list[np.ndarray] = []
    done_paths: list[str] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = []
            valid_paths = []
            for p in batch_paths:
                img = load_rgb(str(p))
                if img is None:
                    continue
                imgs.append(transform(img))
                valid_paths.append(str(p))
            if not imgs:
                continue
            x = torch.stack(imgs).to(device)
            f = model(x)
            feats.append(f.cpu().numpy().astype(np.float32))
            done_paths.extend(valid_paths)

    features = np.vstack(feats) if feats else np.zeros((0, getattr(model, "dim", 0) or 0), np.float32)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, features=features, paths=np.array(done_paths))
    return features, done_paths


def load_rgb(path: str):
    """读图返回 PIL RGB 图（支持中文路径）。失败返回 None。"""
    import cv2
    from PIL import Image
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
