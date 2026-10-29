"""
两阶段端到端自动标注 + 疾病分类。

  阶段 1（检测）：对训练集外的图片自动标注，输出 box + annotation（皮损类别）。
  阶段 2（分类）：由阶段 1 输出的皮损统计直方图 → 疾病分类。

对训练集外的图片，只需要图片本身即可输出：
  - 每个病灶的 box（边界框）与 annotation（皮损类别）
  - 疾病分类结果（湿疹 / 大疱型类天疱疮 / 荨麻疹）
  - （可选）可视化标注图 + 导出 LabelMe 兼容的 .json 标注

用法:
    python -m inference.auto_annotate --image D:/test.jpg
    python -m inference.auto_annotate --image_dir D:/test_images/
    python -m inference.auto_annotate --image D:/test.jpg --conf_threshold 0.3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    DISEASE_CLASSES, DISEASE_CLASSES_ZH,
    LESION_CLASSES, LESION_CLASSES_ZH,
    CHECKPOINT_DIR,
)
from models.multi_task_model import build_model
from models.annotation_stat_classifier import build_annotation_classifier
from data.transforms import get_val_transforms
from data.annotation_features import labels_to_histogram, histogram_to_features

LESION_COLORS = [
    (255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50),
    (255, 50, 255), (50, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 128, 255), (255, 128, 128), (128, 255, 128),
]


class AutoAnnotator:
    """两阶段自动标注器：检测（box + annotation）→ 统计 → 疾病分类。"""

    def __init__(
        self,
        detection_checkpoint: str = None,
        classifier_checkpoint: str = None,
        backbone: str = "convnext_tiny",
        image_size: int = 256,
        conf_threshold: float = 0.3,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold

        # ---- 阶段 1：检测模型 ----
        if detection_checkpoint is None:
            detection_checkpoint = CHECKPOINT_DIR / "best.pth"
        detection_checkpoint = Path(detection_checkpoint)
        if not detection_checkpoint.exists():
            raise FileNotFoundError(
                f"Detection checkpoint not found: {detection_checkpoint}\n"
                f"Train with: python train_labeled.py")

        det_ckpt = torch.load(detection_checkpoint, map_location=self.device,
                              weights_only=False)
        det_cfg = det_ckpt.get("config", {})
        self.det_model = build_model(
            backbone_name=det_cfg.get("backbone", backbone),
            pretrained=False,
            detection_enabled=True,
            detection_head=det_cfg.get("detection_head", "fcos"),
        )
        self.det_model.load_state_dict(det_ckpt["model_state_dict"])
        self.det_model.to(self.device).eval()

        # 图像尺寸优先取 checkpoint 里的配置
        cfg = det_ckpt.get("config", {})
        img_sz = cfg.get("image_size", (image_size, image_size))
        self.image_size = tuple(img_sz) if isinstance(img_sz, (tuple, list)) \
            else (img_sz, img_sz)
        self.transform = get_val_transforms(self.image_size)

        # ---- 阶段 2：皮损统计 → 疾病分类器 ----
        if classifier_checkpoint is None:
            classifier_checkpoint = CHECKPOINT_DIR / "annotation_cls_best.pth"
        classifier_checkpoint = Path(classifier_checkpoint)
        if not classifier_checkpoint.exists():
            raise FileNotFoundError(
                f"Annotation classifier not found: {classifier_checkpoint}\n"
                f"Train with: python train_annotation_classifier.py")
        cls_ckpt = torch.load(classifier_checkpoint, map_location=self.device,
                              weights_only=False)
        cls_cfg = cls_ckpt.get("config", {})
        self.feature_key = cls_cfg.get("feature", "proportion")
        self.cls_model = build_annotation_classifier(
            hidden_dim=cls_cfg.get("hidden_dim", 64),
            dropout=cls_cfg.get("dropout", 0.3),
        )
        self.cls_model.load_state_dict(cls_ckpt["model_state_dict"])
        self.cls_model.to(self.device).eval()

        print(f"Detection checkpoint: {detection_checkpoint} "
              f"(epoch {det_ckpt.get('epoch', '?')}, img {self.image_size})")
        print(f"Classifier checkpoint: {classifier_checkpoint} "
              f"(feature={self.feature_key}, F1 {cls_ckpt.get('best_val_f1', 'N/A'):.4f})")

    @torch.no_grad()
    def annotate(self, image_path: str) -> dict:
        """对单张图片做两阶段自动标注 + 疾病分类。"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 阶段 1：检测 → boxes + annotation（在输入尺寸空间）
        t = self.transform(image=img_rgb, bboxes=[], class_labels=[])
        inp = t["image"].unsqueeze(0).to(self.device)
        out = self.det_model(inp)
        det = out.get("detection", {})
        boxes = det.get("boxes", [])
        labels = det.get("labels", [])
        scores = det.get("scores", [])

        detections = []
        if boxes and len(boxes) > 0 and len(boxes[0]) > 0:
            bb = boxes[0].cpu().numpy()
            lb = labels[0].cpu().numpy() if labels and len(labels) > 0 else None
            sc = scores[0].cpu().numpy() if scores and len(scores) > 0 else None
            in_h, in_w = self.image_size
            for i in range(len(bb)):
                score = float(sc[i]) if sc is not None else 1.0
                if score < self.conf_threshold:
                    continue
                x1 = int(bb[i][0] * orig_w / in_w)
                y1 = int(bb[i][1] * orig_h / in_h)
                x2 = int(bb[i][2] * orig_w / in_w)
                y2 = int(bb[i][3] * orig_h / in_h)
                li = int(lb[i]) if lb is not None else 0
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "label_idx": li,
                    "label": LESION_CLASSES[li] if 0 <= li < len(LESION_CLASSES)
                             else f"class_{li}",
                    "label_zh": LESION_CLASSES_ZH.get(
                        LESION_CLASSES[li], str(li)) if 0 <= li < len(LESION_CLASSES)
                        else str(li),
                    "score": score,
                })

        # 阶段 2：由检测到的 annotation 构建统计直方图 → 疾病分类
        counts = labels_to_histogram(
            [d["label_idx"] for d in detections],
            num_classes=len(LESION_CLASSES),
            scores=[d["score"] for d in detections],
        )
        feats = histogram_to_features(counts)
        hist_t = torch.tensor(feats[self.feature_key], dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        logits = self.cls_model(hist_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())

        return {
            "disease": {
                "class": DISEASE_CLASSES[pred],
                "name_zh": DISEASE_CLASSES_ZH.get(DISEASE_CLASSES[pred], ""),
                "confidence": float(probs[pred]),
                "probs": {DISEASE_CLASSES[i]: float(probs[i])
                          for i in range(len(DISEASE_CLASSES))},
            },
            "lesion_histogram": {LESION_CLASSES[i]: float(counts[i])
                                 for i in range(len(LESION_CLASSES))},
            "detections": detections,
            "image_size": (orig_h, orig_w),
            "image_path": str(image_path),
        }

    def annotate_and_save(self, image_path, output_dir=None, save_json=True,
                          visualize=True) -> dict:
        """推理并（可选）保存可视化图 + LabelMe 兼容 .json。"""
        result = self.annotate(image_path)
        image_path = Path(image_path)
        if output_dir is None:
            output_dir = image_path.parent / "predictions"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if visualize:
            img_bgr = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            vis = img_bgr.copy()
            for det in result["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                color = LESION_COLORS[det["label_idx"] % len(LESION_COLORS)]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                text = f"{det['label_zh']} ({det['score']:.2f})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(vis, text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            d = result["disease"]
            cv2.putText(vis, f"Disease: {d['name_zh']} ({d['confidence']:.2%})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            vis_path = output_dir / f"{image_path.stem}_pred{image_path.suffix}"
            cv2.imencode(image_path.suffix, vis)[1].tofile(str(vis_path))
            print(f"Visualization saved: {vis_path}")

        if save_json:
            shapes = []
            for det in result["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                shapes.append({
                    "label": det["label"],
                    "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                })
            annotation = {
                "version": "5.5.0",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_path.name,
                "imageData": None,
                "imageHeight": result["image_size"][0],
                "imageWidth": result["image_size"][1],
            }
            json_path = output_dir / f"{image_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)
            print(f"Annotation saved: {json_path}")

        return result


def main():
    p = argparse.ArgumentParser(
        description="Two-stage auto-annotation + disease classification")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--detection_checkpoint", type=str, default=None)
    p.add_argument("--classifier_checkpoint", type=str, default=None)
    p.add_argument("--backbone", type=str, default="convnext_tiny")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--conf_threshold", type=float, default=0.3)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--no_json", action="store_true")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if not args.image and not args.image_dir:
        p.error("Either --image or --image_dir must be specified")

    annotator = AutoAnnotator(
        detection_checkpoint=args.detection_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        backbone=args.backbone,
        image_size=args.image_size,
        conf_threshold=args.conf_threshold,
        device=args.device,
    )

    if args.image:
        result = annotator.annotate_and_save(
            args.image, output_dir=args.output_dir, save_json=not args.no_json)
        d = result["disease"]
        print(f"\n{'='*50}")
        print(f"IMAGE: {Path(args.image).name}")
        print(f"{'='*50}")
        print(f"Disease: {d['name_zh']} ({d['class']}) — {d['confidence']:.2%}")
        print("Probabilities:")
        for c, pr in sorted(d["probs"].items(), key=lambda x: -x[1]):
            print(f"  {DISEASE_CLASSES_ZH.get(c, c)} ({c}): {pr:.4f}")
        print(f"\nAnnotations ({len(result['detections'])} boxes):")
        for i, det in enumerate(result["detections"]):
            print(f"  [{i+1}] box={det['bbox']} annotation={det['label_zh']} "
                  f"({det['label']}) score={det['score']:.3f}")
        nz = {k: v for k, v in result["lesion_histogram"].items() if v > 0}
        print(f"\nLesion histogram (annotation statistics): {nz}")

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        files = sorted(f for f in image_dir.iterdir() if f.suffix in exts)
        for i, f in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] {f.name}")
            try:
                r = annotator.annotate_and_save(
                    str(f), output_dir=args.output_dir, save_json=not args.no_json)
                print(f"  Disease: {r['disease']['name_zh']} "
                      f"({r['disease']['confidence']:.2%}) | "
                      f"{len(r['detections'])} boxes")
            except Exception as e:
                print(f"  [ERROR] {e}")


if __name__ == "__main__":
    main()
