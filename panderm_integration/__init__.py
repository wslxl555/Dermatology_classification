"""
panderm_integration：PanDerm 权重就绪后，把它的 ViT 特征用于
(1) linear probe（linear_probe.py）与 (2) 检测骨干接入（detector_backbone.py）。

CLI 模块（linear_probe / detector_backbone）按需惰性导入，避免
`python -m panderm_integration.xxx` 时的重复导入告警。
"""

from . import model
from . import features

__all__ = ["model", "features", "linear_probe", "detector_backbone"]


def __getattr__(name):
    if name in ("linear_probe", "detector_backbone"):
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
