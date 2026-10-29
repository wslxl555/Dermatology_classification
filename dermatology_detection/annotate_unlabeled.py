"""
对 unlabeled/ 下的 784 张图片运行检测器，输出三份标注产物：

  1. LabelMe JSON —— 每张图一个 .json（bbox + 皮损类型 + 疾病真值）
  2. 可视化 JPG —— 在原图上绘制彩色检测框 + 中文皮损标签 + 疾病标签
  3. CSV 汇总 —— 每个检测框一行

疾病真值取自图片所在目录名（湿疹 / 类天疱疹 / 荨麻疹）。
皮损类型来自 15 类 LESION_CLASSES。

用法:
    python annotate_unlabeled.py --checkpoint best.pth
    python annotate_unlabeled.py --checkpoint best.pth --conf_threshold 0.3
    python annotate_unlabeled.py --checkpoint best.pth --max_images 10
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR,
    DISEASE_CLASSES, DISEASE_CLASSES_ZH,
    LESION_CLASSES, LESION_CLASSES_ZH,
)
from data.prepare_labeled_data import scan_labeled_dataset
from data.transforms import get_cls_val_transforms
from models.multi_task_model import build_model


# 可视化颜色（每类一个）
LESION_COLORS = [
    (255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50),
    (255, 50, 255), (50, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 128, 255), (255, 128, 128), (128, 255, 128),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate unlabeled images with lesion boxes + types")
    parser.add_argument("--checkpoint", type=str, default="best.pth",
                        help="Checkpoint file name or absolute path")
    parser.add_argument("--unlabeled_root", type=str, default=None,
                        help="Path to unlabeled/ (default: ../unlabeled)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: outputs/unlabeled_annotations/)")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images (for a quick test)")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


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
    in_h, in_w = img_sz

    if not detection_enabled:
        raise RuntimeError("Checkpoint has detection_enabled=False; cannot annotate boxes.")

    # 扫描 unlabeled
    unlabeled_root = args.unlabeled_root
    if unlabeled_root is None:
        unlabeled_root = _project_root.parent / "unlabeled"
    unlabeled_root = Path(unlabeled_root).resolve()
    print(f"Unlabeled root: {unlabeled_root}")
    samples = scan_labeled_dataset(unlabeled_root)
    if not samples:
        raise RuntimeError(f"No images found under {unlabeled_root}")
    if args.max_images is not None:
        samples = samples[:args.max_images]
    print(f"Images to annotate: {len(samples)}")

    # 构建模型
    model = build_model(
        backbone_name=backbone, pretrained=False,
        detection_enabled=True, detection_head=detection_head,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded: backbone={backbone}, detection_head={detection_head}, "
          f"image_size={img_sz}")

    transform = get_cls_val_transforms(img_sz)

    # 输出目录（按疾病子目录镜像）
    out_root = Path(args.output_dir) if args.output_dir else (OUTPUT_DIR / "unlabeled_annotations")
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "annotations.csv"
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["image_id", "disease", "disease_cn", "x1", "y1", "x2", "y2",
                     "lesion_label", "lesion_zh", "score"])

    n_det_total = 0
    n_processed = 0
    for idx, sample in enumerate(samples):
        image_path = Path(sample["image_path"])
        disease_en = sample["disease"]
        disease_cn = DISEASE_CLASSES_ZH.get(disease_en, disease_en)

        # 读取原图
        img_bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [SKIP] failed to load {image_path}")
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 预处理（仅 resize + normalize）
        transformed = transform(image=img_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        # 推理
        output = model(input_tensor)

        # 疾病分类（记录模型预测，但真值用目录名）
        disease_probs = F.softmax(output["disease_logits"], dim=-1).cpu().numpy()[0]
        disease_pred = int(np.argmax(disease_probs))

        # 检测
        detections = []
        det_output = output.get("detection", {})
        boxes = det_output.get("boxes", [])
        labels = det_output.get("labels", [])
        scores = det_output.get("scores", [])
        if boxes and len(boxes) > 0 and boxes[0].numel() > 0:
            bb = boxes[0].cpu().numpy()
            ll = labels[0].cpu().numpy()
            ss = scores[0].cpu().numpy()
            for i in range(len(bb)):
                score = float(ss[i])
                if score < args.conf_threshold:
                    continue
                x1, y1, x2, y2 = bb[i]
                # 映射回原图分辨率
                x1 = int(x1 * orig_w / in_w)
                y1 = int(y1 * orig_h / in_h)
                x2 = int(x2 * orig_w / in_w)
                y2 = int(y2 * orig_h / in_h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                label_idx = int(ll[i])
                if 0 <= label_idx < len(LESION_CLASSES):
                    lesion_name = LESION_CLASSES[label_idx]
                    lesion_zh = LESION_CLASSES_ZH.get(lesion_name, lesion_name)
                else:
                    lesion_name = f"class_{label_idx}"
                    lesion_zh = lesion_name
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": lesion_name,
                    "label_zh": lesion_zh,
                    "score": score,
                })

        # 保存产物（按疾病子目录镜像）
        sub_out = out_root / disease_cn
        sub_out.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        # 1) LabelMe JSON
        shapes = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            shapes.append({
                "label": det["label"],
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            })
        annotation = {
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": orig_h,
            "imageWidth": orig_w,
            "disease": disease_en,
            "disease_zh": disease_cn,
        }
        json_path = sub_out / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        # 2) 可视化 JPG
        vis = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            c_idx = LESION_CLASSES.index(det["label"]) if det["label"] in LESION_CLASSES else 0
            color = LESION_COLORS[c_idx % len(LESION_COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"{det['label_zh']} {det['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, text, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        header = f"{disease_cn} (pred: {DISEASE_CLASSES_ZH.get(DISEASE_CLASSES[disease_pred], '')})"
        cv2.putText(vis, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        vis_path = sub_out / f"{stem}_annotated{image_path.suffix}"
        cv2.imencode(image_path.suffix, vis)[1].tofile(str(vis_path))

        # 3) CSV 行
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            writer.writerow([stem, disease_en, disease_cn, x1, y1, x2, y2,
                             det["label"], det["label_zh"], f"{det['score']:.4f}"])
        n_det_total += len(detections)
        n_processed += 1

        if (idx + 1) % 20 == 0 or (idx + 1) == len(samples):
            print(f"  [{idx+1}/{len(samples)}] processed, "
                  f"cumulative detections={n_det_total}")

    csv_f.close()
    print(f"\n{'='*50}")
    print(f"ANNOTATION COMPLETE: {n_processed}/{len(samples)} images, "
          f"{n_det_total} detections")
    print(f"  JSON + JPG under: {out_root}")
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
