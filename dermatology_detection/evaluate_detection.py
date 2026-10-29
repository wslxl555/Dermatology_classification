"""
在 labeled 验证集上评估检测分支的 box mAP。

从 checkpoint 恢复 backbone + detection_head，在 val 划分上运行推理，
用 GT bbox 计算 COCO 风格 mAP@0.5 / mAP@0.75 / mAP@[.5,.75]。

用法:
    python evaluate_detection.py --checkpoint best.pth
    python evaluate_detection.py --checkpoint best.pth --split labeled_splits.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR, NUM_LESION_CLASSES, LESION_CLASSES,
)
from data.dataset import EczemaDataset, _resolve_path
from data.label_mapping import parse_label, LESION_CLASS_TO_IDX
from data.transforms import get_val_transforms
from models.multi_task_model import build_model
from training.det_metrics import compute_map
from utils.labelme_parser import parse_labelme
from utils.voc_parser import parse_voc_xml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate detection mAP on labeled val split")
    parser.add_argument("--checkpoint", type=str, default="best.pth",
                        help="Checkpoint file name or absolute path")
    parser.add_argument("--split", type=str, default="labeled_splits_trainval.pkl",
                        help="Splits pkl under outputs/ (trainval or original)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--conf_threshold", type=float, default=0.05,
                        help="Score threshold applied to detections before mAP")
    parser.add_argument("--tile", action="store_true",
                        help="滑窗切块推理:在原生分辨率上滑窗预测并拼接,"
                             "用于评估以 --augmentation tile 训练的模型")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="滑窗块大小(默认 512,需与训练 image_size 一致)")
    parser.add_argument("--tile_stride", type=int, default=256,
                        help="滑窗步长(默认 256,与 tile_size 重叠一半)")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    bboxes = [b["bboxes"] for b in batch]
    lesion_labels = [b["lesion_labels"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    return {"image": images, "bboxes": bboxes,
            "lesion_labels": lesion_labels, "image_id": image_ids}


# ---------------------------------------------------------------------------
# 滑窗切块推理(与 --augmentation tile 训练配套)
# ---------------------------------------------------------------------------

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_patch(patch_rgb: np.ndarray) -> torch.Tensor:
    """把 [H, W, 3] uint8 RGB 块转成 [1, 3, H, W] 归一化张量。"""
    x = patch_rgb.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()
    return x.unsqueeze(0)


def _load_native_gt(sample: dict):
    """读取样本原生分辨率下的 GT 框与类别索引(含复合标签展开)。"""
    boxes, labels = [], []
    json_path = _resolve_path(sample.get("json_path"))
    xml_path = _resolve_path(sample.get("xml_path"))
    anns = []
    if json_path and Path(json_path).exists():
        parsed = parse_labelme(Path(json_path))
        anns += (parsed.get("annotations") or [])
    if xml_path and Path(xml_path).exists():
        parsed = parse_voc_xml(Path(xml_path))
        anns += (parsed.get("annotations") or [])

    for ann in anns:
        bbox = ann["bbox"]  # [x1, y1, x2, y2] 原生坐标
        for cls_name in parse_label(ann["label"]):
            idx = LESION_CLASS_TO_IDX.get(cls_name)
            if idx is not None:
                boxes.append(bbox)
                labels.append(idx)

    if boxes:
        return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)
    return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)


@torch.no_grad()
def _predict_tiled(model, image_rgb: np.ndarray, tile_size: int, stride: int,
                   device: str, conf_thresh: float = 0.05):
    """滑窗推理:把整图切成 tile_size 块,逐块预测并拼接,再全局 NMS。

    返回 (boxes, scores, labels) 张量,坐标在整图原生分辨率下。
    """
    H, W = image_rgb.shape[:2]

    if H > tile_size:
        ys = list(range(0, H - tile_size + 1, stride))
        if ys[-1] + tile_size < H:
            ys.append(H - tile_size)
    else:
        ys = [0]
    if W > tile_size:
        xs = list(range(0, W - tile_size + 1, stride))
        if xs[-1] + tile_size < W:
            xs.append(W - tile_size)
    else:
        xs = [0]

    all_boxes, all_scores, all_labels = [], [], []
    for y0 in ys:
        for x0 in xs:
            patch = image_rgb[y0:y0 + tile_size, x0:x0 + tile_size]
            patch_t = _preprocess_patch(patch).to(device)
            out = model(patch_t, targets=None)
            det = out.get("detection", {})
            box_list = det.get("boxes") or []
            boxes = box_list[0] if box_list else None
            score_list = det.get("scores") or []
            scores = score_list[0] if score_list else None
            label_list = det.get("labels") or []
            labels = label_list[0] if label_list else None
            if boxes is None or boxes.numel() == 0:
                continue

            keep = scores >= conf_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if boxes.numel() == 0:
                continue

            off = torch.tensor([x0, y0, x0, y0], device=boxes.device,
                               dtype=boxes.dtype)
            all_boxes.append(boxes + off)
            all_scores.append(scores)
            all_labels.append(labels)

    if not all_boxes:
        return (torch.zeros((0, 4)), torch.zeros(0),
                torch.zeros(0, dtype=torch.long))
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
    return boxes[keep].cpu(), scores[keep].cpu(), labels[keep].cpu()


@torch.no_grad()
def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = CHECKPOINT_DIR / args.checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    backbone = cfg.get("backbone", "convnext_tiny")
    detection_head = cfg.get("detection_head", "fcos")
    detection_enabled = cfg.get("detection_enabled", True)
    img_sz = cfg.get("image_size", (512, 512))
    img_sz = tuple(img_sz) if isinstance(img_sz, (tuple, list)) else (img_sz, img_sz)

    if not detection_enabled:
        raise RuntimeError("Checkpoint was trained with detection_enabled=False; "
                           "cannot evaluate detection mAP.")

    # 加载 val 划分
    splits_path = OUTPUT_DIR / args.split
    print(f"Loading splits: {splits_path}")
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)
    val_samples = splits.get("val", [])
    if not val_samples:
        raise RuntimeError(f"No 'val' samples found in {splits_path}")
    print(f"Val samples: {len(val_samples)}")

    # 构建模型并加载权重
    model = build_model(
        backbone_name=backbone,
        pretrained=False,
        detection_enabled=True,
        detection_head=detection_head,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded: backbone={backbone}, detection_head={detection_head}, "
          f"image_size={img_sz}")

    predictions, ground_truths = [], []
    if args.tile:
        # 滑窗切块评估:原生分辨率逐块推理并拼接
        for s in val_samples:
            image_path = _resolve_path(s["image_path"])
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            b, sc, l = _predict_tiled(model, img, args.tile_size, args.tile_stride,
                                      device, args.conf_threshold)
            predictions.append({
                "boxes": b.numpy(), "scores": sc.numpy(), "labels": l.numpy(),
            })
            gt_b, gt_l = _load_native_gt(s)
            ground_truths.append({"boxes": gt_b, "labels": gt_l})
    else:
        transform = get_val_transforms(img_sz)
        dataset = EczemaDataset(val_samples, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=_collate_fn)

        for batch in loader:
            images = batch["image"].to(device)
            output = model(images, targets=None)
            det = output.get("detection", {})
            boxes = det.get("boxes", [])
            labels = det.get("labels", [])
            scores = det.get("scores", [])

            for i in range(images.shape[0]):
                # 预测（模型输入空间，与 GT 已 resize 到同一空间）
                b = boxes[i].cpu()
                s = scores[i].cpu()
                l = labels[i].cpu()
                if b.numel() > 0:
                    keep = s >= args.conf_threshold
                    b, s, l = b[keep], s[keep], l[keep]
                    # 按分数降序（compute_map 内部也会排序，但这里保证顺序一致）
                    idx = torch.argsort(s, descending=True)
                    b, s, l = b[idx], s[idx], l[idx]
                predictions.append({
                    "boxes": b.numpy(), "scores": s.numpy(), "labels": l.numpy(),
                })

                gt_b = batch["bboxes"][i].cpu()
                gt_l = batch["lesion_labels"][i].cpu()
                ground_truths.append({
                    "boxes": gt_b.numpy(), "labels": gt_l.numpy(),
                })

    # 计算 mAP
    results = compute_map(predictions, ground_truths, NUM_LESION_CLASSES,
                          iou_thresholds=(0.5, 0.75))

    print("\n" + "=" * 60)
    print("DETECTION mAP ON LABELED VAL SPLIT")
    print("=" * 60)
    print(f"  mAP@0.5:       {results['mAP@0.5']:.4f}")
    print(f"  mAP@0.75:      {results['mAP@0.75']:.4f}")
    print(f"  mAP@[.5,.75]:  {results['mAP@[.5,.75]']:.4f}")
    print("\nPer-class AP@0.5:")
    for c, name in enumerate(LESION_CLASSES):
        print(f"  {name:18s}: {results.get(f'AP_cls{c}@0.5', 0.0):.4f}")

    # 保存 JSON
    import json
    out = {
        "checkpoint": str(ckpt_path),
        "backbone": backbone,
        "detection_head": detection_head,
        "image_size": list(img_sz),
        "num_val_samples": len(val_samples),
        "conf_threshold": args.conf_threshold,
        "results": results,
    }
    out_path = OUTPUT_DIR / "evaluation_detection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
