"""Smoke test: build every detection head, run train + inference forward, and
verify output shapes/loss keys. Uses synthetic FPN features so no dataset is
needed (fast, CPU-only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from config import NUM_LESION_CLASSES, MODEL_CONFIG
from models.detection_head import build_detection_head

HEADS = ["fcos", "faster_rcnn", "cascade_rcnn", "yolov8", "rtdetr",
         "transunet", "transcascade", "attention_unet", "mamba", "sam_adapter"]
IN_CHANNELS = [96, 192, 384, 768]  # convnext_tiny stage channels
FPN = MODEL_CONFIG["fpn_channels"]  # 256


def make_features(B=2, H=256, W=256):
    """Synthetic pyramid features at strides 4/8/16/32."""
    feats = {}
    for i, c in enumerate(IN_CHANNELS):
        s = 2 ** (i + 2)
        feats[f"stage{i+1}"] = torch.randn(B, c, H // s, W // s)
    return feats


def make_targets(B=2, H=256, W=256):
    targets = []
    for b in range(B):
        # 2 GT boxes per image, deterministic + non-degenerate
        boxes = torch.tensor([
            [40.0, 50.0, 140.0, 160.0],
            [120.0, 130.0, 220.0, 240.0],
        ])
        labels = torch.tensor([1, 4], dtype=torch.long)
        targets.append({"boxes": boxes, "labels": labels})
    return targets


def main():
    B, H, W = 2, 256, 256
    features = make_features(B, H, W)
    images = torch.randn(B, 3, H, W)
    targets = make_targets(B, H, W)

    ok = True
    for name in HEADS:
        print(f"\n=== {name} ===")
        try:
            head = build_detection_head(name, IN_CHANNELS, FPN, NUM_LESION_CLASSES)
            n = sum(p.numel() for p in head.parameters())
            print(f"  built: {n/1e6:.2f}M params")

            # train forward
            out = head(features, images, targets)
            losses = out.get("losses", {})
            assert losses, "no losses returned in train mode"
            print(f"  train losses: { {k: round(float(v), 4) for k, v in losses.items()} }")

            # backward
            total = sum(v for v in losses.values())
            total.backward()
            print("  backward: ok")

            # inference forward
            head.eval()
            with torch.no_grad():
                out = head(features, images, None)
            boxes = out["boxes"]
            labels = out["labels"]
            scores = out["scores"]
            assert len(boxes) == B, f"expected {B} boxes lists, got {len(boxes)}"
            assert len(labels) == B and len(scores) == B
            for bb in boxes:
                assert bb.shape[-1] == 4, f"boxes should be [N,4], got {bb.shape}"
            print(f"  inference detections per image: {[len(b) for b in boxes]}")

            # lesion feature
            with torch.no_grad():
                lf = head.get_lesion_features(features, images)
            assert lf.shape[0] == B, f"lesion features wrong shape {lf.shape}"
            print(f"  lesion_features: {tuple(lf.shape)}")

            print(f"  PASSED")
        except Exception as e:
            ok = False
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")

    print("\n" + ("ALL HEADS PASSED" if ok else "SOME HEADS FAILED"))


if __name__ == "__main__":
    main()
