"""
独立病灶检测脚本（供可解释性推理子进程调用）。

加载多任务模型（含 sam_adapter 等分割式检测头），对一张图或一个文件夹跑检测，
把结果以 JSON 输出（原图坐标下的 bbox + 类别名 + 置信度）。

设计原因：dermatology_classifier 与 dermatology_detection 两个工程都有顶层
config.py / models/ 模块，直接 import 会发生模块名冲突，因此这里作为独立脚本
被子进程调用（与 detector/train_segmentation.py 桥接训练是同一模式）。

用法:
    python inference/detect_lesions.py --checkpoint outputs/checkpoints/best.pth --dir /path --out det.json
    python inference/detect_lesions.py --checkpoint best.pth --image a.jpg --conf_threshold 0.3
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import LESION_CLASSES, CHECKPOINT_DIR  # noqa: E402
from models.multi_task_model import build_model  # noqa: E402
from data.transforms import get_val_transforms  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_model(checkpoint_path: str, device: str):
    """加载多任务 checkpoint，返回 (model, transform, image_size)。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    backbone = cfg.get("backbone", "convnext_tiny")
    detection_head = cfg.get("detection_head", "sam_adapter")
    detection_enabled = cfg.get("detection_enabled", True)

    image_size = cfg.get("image_size", (512, 512))
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    image_size = tuple(int(x) for x in image_size)

    model = build_model(
        backbone_name=backbone,
        pretrained=False,
        detection_enabled=detection_enabled,
        detection_head=detection_head,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    transform = get_val_transforms(image_size=image_size, level="standard")
    return model, transform, image_size


@torch.no_grad()
def detect(model, transform, image_size, img_rgb, device, conf_thresh):
    """对一张 RGB 图跑检测，返回 [{bbox, label, score}, ...]（原图坐标）。"""
    t = transform(image=img_rgb, bboxes=[], class_labels=[])
    inp = t["image"].unsqueeze(0).to(device)
    out = model(inp)

    det = out.get("detection", {}) or {}
    boxes = det.get("boxes") or []
    labels = det.get("labels") or []
    scores = det.get("scores") or []

    if not boxes or len(boxes) == 0 or boxes[0].numel() == 0:
        return []

    bb = boxes[0].cpu().numpy()
    ll = labels[0].cpu().numpy()
    ss = scores[0].cpu().numpy()

    in_h, in_w = image_size
    orig_h, orig_w = img_rgb.shape[:2]

    results = []
    for i in range(len(bb)):
        sc = float(ss[i])
        if sc < conf_thresh:
            continue
        x1 = int(round(bb[i][0] * orig_w / in_w))
        y1 = int(round(bb[i][1] * orig_h / in_h))
        x2 = int(round(bb[i][2] * orig_w / in_w))
        y2 = int(round(bb[i][3] * orig_h / in_h))
        li = int(ll[i])
        name = LESION_CLASSES[li] if 0 <= li < len(LESION_CLASSES) else "other"
        results.append({"bbox": [x1, y1, x2, y2], "label": name, "score": sc})
    return results


def main():
    ap = argparse.ArgumentParser(description="独立病灶检测（输出 JSON）")
    ap.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best.pth"))
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--conf_threshold", type=float, default=0.05)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, image_size = load_model(args.checkpoint, device)

    if args.image:
        paths = [Path(args.image)]
    elif args.dir:
        paths = sorted(
            p for p in Path(args.dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    else:
        print(json.dumps({}), file=sys.stderr)
        sys.exit(1)

    out = {}
    for p in paths:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            out[p.name] = []
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out[p.name] = detect(model, transform, image_size, img_rgb, device,
                             args.conf_threshold)

    payload = json.dumps(out, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
