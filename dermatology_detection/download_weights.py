"""
将预训练 backbone 权重预下载到本地 torch 缓存。

请在提交训练任务到计算节点之前,在登录节点(有网络访问权限)上运行一次本脚本。
HPC 主目录是共享的,因此计算节点无需联网即可找到已缓存的文件。

用法:
    python download_weights.py                  # 缓存所有支持的 backbone
    python download_weights.py --backbone convnext_tiny   # 只缓存某一个
"""

import argparse
import os
import sys
from pathlib import Path

# ── 项目可能用到的 backbone ──────────────────────────────────────────
BACKBONE_WEIGHTS = {
    "convnext_tiny": {
        "url": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        "pkg": "torchvision.models",
        "builder": "convnext_tiny",
    },
    "convnext_small": {
        "url": "https://download.pytorch.org/models/convnext_small-8fbaf13d.pth",
        "pkg": "torchvision.models",
        "builder": "convnext_small",
    },
    "convnext_base": {
        "url": "https://download.pytorch.org/models/convnext_base-6075fb77.pth",
        "pkg": "torchvision.models",
        "builder": "convnext_base",
    },
    "swin_t": {
        "url": "https://download.pytorch.org/models/swin_t-704ceda3.pth",
        "pkg": "torchvision.models",
        "builder": "swin_t",
    },
    "swin_s": {
        "url": "https://download.pytorch.org/models/swin_s-8b1d03a0.pth",
        "pkg": "torchvision.models",
        "builder": "swin_s",
    },
    "resnet50": {
        "url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "pkg": "torchvision.models",
        "builder": "resnet50",
    },
    "resnet101": {
        "url": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
        "pkg": "torchvision.models",
        "builder": "resnet101",
    },
}


def get_cache_dir() -> Path:
    """返回 torch hub 的 checkpoints 目录。"""
    hub_dir = os.environ.get(
        "TORCH_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub"),
    )
    return Path(hub_dir) / "checkpoints"


def download_via_url(url: str, dst: Path) -> bool:
    """通过 urllib(标准库,无额外依赖)下载文件。"""
    from urllib.request import urlretrieve

    if dst.exists():
        print(f"  ✓ already cached: {dst}")
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} …")
    try:
        urlretrieve(url, dst)
        print(f"  ✓ saved to {dst}")
        return True
    except Exception as e:
        print(f"  ✗ failed: {e}", file=sys.stderr)
        return False


def download_via_torchvision(name: str) -> bool:
    """
    让 torchvision 自带的 API 下载并缓存权重。
    这是最可靠的方式,因为它还会校验哈希值。
    """
    from torchvision.models import (
        convnext_tiny, convnext_small, convnext_base,
        swin_t, swin_s,
        resnet50, resnet101,
        ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights,
        Swin_T_Weights, Swin_S_Weights,
        ResNet50_Weights, ResNet101_Weights,
    )

    BUILDERS = {
        "convnext_tiny":  (convnext_tiny,  ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "convnext_small": (convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1),
        "convnext_base":  (convnext_base,  ConvNeXt_Base_Weights.IMAGENET1K_V1),
        "swin_t":         (swin_t,         Swin_T_Weights.IMAGENET1K_V1),
        "swin_s":         (swin_s,         Swin_S_Weights.IMAGENET1K_V1),
        "resnet50":       (resnet50,       ResNet50_Weights.IMAGENET1K_V1),
        "resnet101":      (resnet101,      ResNet101_Weights.IMAGENET1K_V1),
    }

    if name not in BUILDERS:
        print(f"  ✗ unknown backbone: {name}", file=sys.stderr)
        return False

    builder, weights_enum = BUILDERS[name]
    print(f"  loading '{name}' with pretrained weights (torchvision will cache) …")
    try:
        builder(weights=weights_enum)
        print(f"  ✓ '{name}' cached successfully")
        return True
    except Exception as e:
        print(f"  ✗ failed: {e}", file=sys.stderr)
        return False


def main():
    """脚本主入口:下载全部或指定的 backbone 权重。"""
    parser = argparse.ArgumentParser(
        description="Pre-download pretrained backbone weights for offline HPC compute nodes"
    )
    parser.add_argument(
        "--backbone", type=str, default=None,
        choices=list(BACKBONE_WEIGHTS.keys()),
        help="Download a single backbone (default: all)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Override torch hub cache directory",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else get_cache_dir()
    print(f"Torch hub cache: {cache_dir}")
    print()

    names = [args.backbone] if args.backbone else list(BACKBONE_WEIGHTS.keys())
    ok = 0
    fail = 0

    for name in names:
        print(f"[{name}]")
        # 先尝试 torchvision(会校验哈希),失败后再直接通过 URL 下载
        success = download_via_torchvision(name)
        if not success:
            info = BACKBONE_WEIGHTS[name]
            dst = get_cache_dir() / Path(info["url"]).name
            print(f"  torchvision failed — trying direct URL download …")
            success = download_via_url(info["url"], dst)
        if success:
            ok += 1
        else:
            fail += 1
        print()

    print(f"Done: {ok} cached, {fail} failed")
    if fail:
        print("\nTIP: Run this script on the LOGIN NODE, not a compute node.")
        print("If the login node also lacks DNS, download the .pth files from")
        print("a machine with internet access and copy them to:")
        print(f"  {get_cache_dir()}")
        print()
        print("You can also try setting a proxy:")
        print("  export https_proxy=http://proxy-server:port")
        print("  python download_weights.py")
        sys.exit(1)

    print(f"\nYou can now submit training jobs with --offline:")
    print(f"  python train.py --offline --epochs 50")


if __name__ == "__main__":
    main()
