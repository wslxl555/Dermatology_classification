"""
使用训练好的模型对皮肤病图片进行自动检测和标注。

功能:
  - 疾病分类（湿疹/类天疱疮/荨麻疹）
  - 皮损检测（15 类皮损的边界框）
  - 可视化标注图（在原图上绘制检测框）
  - 导出 LabelMe 兼容的 JSON 标注

用法:
    # 预测单张图片
    python -m inference.predict_labeled --image D:/test.jpg

    # 预测整个文件夹
    python -m inference.predict_labeled --image_dir D:/test_images/

    # 指定模型 checkpoint
    python -m inference.predict_labeled --image test.jpg --checkpoint best.pth

    # 输出可视化图和 JSON 到指定目录
    python -m inference.predict_labeled --image test.jpg --output_dir results/

    # 调整置信度阈值
    python -m inference.predict_labeled --image test.jpg --conf_threshold 0.3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# 动态添加项目根目录
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    DISEASE_CLASSES, DISEASE_CLASSES_ZH,
    LESION_CLASSES, LESION_CLASSES_ZH,
    CHECKPOINT_DIR,
)
from models.multi_task_model import build_model
from data.transforms import get_val_transforms
from data.label_mapping import LESION_CLASSES as DET_LESION_CLASSES


# 用于可视化绘制的颜色（每个类别一个颜色）
LESION_COLORS = [
    (255, 50, 50),     # macule - 红
    (50, 255, 50),     # patch - 绿
    (50, 50, 255),     # papule - 蓝
    (255, 255, 50),    # plaque - 黄
    (255, 50, 255),    # nodule - 品红
    (50, 255, 255),    # vesicle - 青
    (255, 128, 0),     # bulla - 橙
    (128, 0, 255),     # pustule - 紫
    (0, 128, 255),     # scale - 天蓝
    (255, 0, 128),     # crust - 玫红
    (128, 255, 0),     # erosion - 黄绿
    (0, 255, 128),     # excoriation - 春绿
    (128, 128, 255),   # scar - 淡紫
    (255, 128, 128),   # lichenification - 粉红
    (128, 255, 128),   # fissure - 淡绿
]


class LesionPredictor:
    """皮肤病自动检测和标注预测器。"""

    def __init__(
        self,
        checkpoint_path: str = None,
        backbone: str = "convnext_tiny",
        detection_enabled: bool = True,
        image_size: tuple = (512, 512),
        conf_threshold: float = 0.5,
        device: str = None,
    ):
        """
        参数:
            checkpoint_path: 训练好的模型 checkpoint 路径
            backbone: 主干网络名称
            detection_enabled: 是否启用检测分支
            image_size: 输入图像尺寸
            conf_threshold: 检测框置信度阈值
            device: 运行设备
        """
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载权重（先读 checkpoint 配置，architecture 以 checkpoint 为准）
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "best.pth"

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Train a model first with: python train_labeled.py"
            )

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        cfg = checkpoint.get("config", {})

        # 检测头/骨干以 checkpoint 记录为准，避免架构不匹配导致加载失败
        detection_head = cfg.get("detection_head", "fcos")
        backbone = cfg.get("backbone", backbone)
        detection_enabled = cfg.get("detection_enabled", detection_enabled)

        # 构建模型
        self.model = build_model(
            backbone_name=backbone,
            pretrained=False,
            detection_enabled=detection_enabled,
            detection_head=detection_head,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.detection_enabled = detection_enabled

        print(f"Model loaded on {self.device}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best val F1: {checkpoint.get('best_val_f1', 'N/A')}")
        print(f"  Detection enabled: {detection_enabled}")

        # 数据预处理
        self.transform = get_val_transforms(image_size=image_size)

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        对单张图片进行推理。

        返回:
            dict: {
                "disease": {"class": str, "name_zh": str, "confidence": float, "probs": list},
                "detections": [{"bbox": [x1,y1,x2,y2], "label": str, "label_zh": str, "score": float}],
                "image_size": (h, w),
            }
        """
        # 加载图像
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")

        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 预处理
        transformed = self.transform(image=img_rgb, bboxes=[], class_labels=[])
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # 推理
        output = self.model(input_tensor)

        # 疾病分类
        disease_probs = F.softmax(output["disease_logits"], dim=-1).cpu().numpy()[0]
        disease_pred = int(np.argmax(disease_probs))

        disease_result = {
            "class": DISEASE_CLASSES[disease_pred],
            "name_zh": DISEASE_CLASSES_ZH.get(DISEASE_CLASSES[disease_pred], ""),
            "confidence": float(disease_probs[disease_pred]),
            "probs": {
                DISEASE_CLASSES[i]: float(disease_probs[i])
                for i in range(len(DISEASE_CLASSES))
            },
        }

        # 皮损检测
        detections = []
        if self.detection_enabled and "detection" in output:
            det_output = output["detection"]
            boxes = det_output.get("boxes", [])
            labels = det_output.get("labels", [])
            scores = det_output.get("scores", [])

            # 模型检测输出为 list[Tensor]（每个 batch 图像一个元素），
            # 单图推理时取第一个元素即可。
            if boxes is not None and len(boxes) > 0:
                batch_boxes = boxes[0]
                batch_labels = labels[0] if labels and len(labels) > 0 else None
                batch_scores = scores[0] if scores and len(scores) > 0 else None

                if isinstance(batch_boxes, torch.Tensor):
                    batch_boxes = batch_boxes.cpu().numpy()
                if isinstance(batch_labels, torch.Tensor):
                    batch_labels = batch_labels.cpu().numpy()
                if isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.cpu().numpy()

                for i in range(len(batch_boxes)):
                    score = float(batch_scores[i]) if (
                        batch_scores is not None and i < len(batch_scores)
                    ) else 1.0
                    if score < self.conf_threshold:
                        continue

                    bbox = batch_boxes[i]
                    # 检测框位于模型输入空间（image_size）中，按比例映射回原图像素坐标
                    in_w = self.image_size[1]
                    in_h = self.image_size[0]
                    x1 = int(bbox[0] * orig_w / in_w)
                    y1 = int(bbox[1] * orig_h / in_h)
                    x2 = int(bbox[2] * orig_w / in_w)
                    y2 = int(bbox[3] * orig_h / in_h)

                    label_idx = int(batch_labels[i])
                    if 0 <= label_idx < len(LESION_CLASSES):
                        lesion_name = LESION_CLASSES[label_idx]
                        lesion_name_zh = LESION_CLASSES_ZH.get(lesion_name, lesion_name)
                    else:
                        lesion_name = f"class_{label_idx}"
                        lesion_name_zh = lesion_name

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "label": lesion_name,
                        "label_zh": lesion_name_zh,
                        "score": score,
                    })

        return {
            "disease": disease_result,
            "detections": detections,
            "image_size": (orig_h, orig_w),
            "image_path": str(image_path),
        }

    def predict_and_visualize(
        self,
        image_path: str,
        output_dir: str = None,
        save_json: bool = True,
    ) -> dict:
        """
        推理并保存可视化结果和 JSON 标注。

        参数:
            image_path: 输入图片路径
            output_dir: 输出目录（默认：图片所在目录下的 predictions/）
            save_json: 是否保存 LabelMe 兼容 JSON

        返回:
            predict() 的结果字典
        """
        result = self.predict(image_path)
        image_path = Path(image_path)

        # 创建输出目录
        if output_dir is None:
            output_dir = image_path.parent / "predictions"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载原图用于可视化
        img_bgr = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        orig_h, orig_w = img_bgr.shape[:2]

        # 在图上绘制检测框
        vis_img = img_bgr.copy()
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            label_idx = LESION_CLASSES.index(det["label"]) if det["label"] in LESION_CLASSES else 0
            color = LESION_COLORS[label_idx % len(LESION_COLORS)]

            # 绘制矩形框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label_text = f"{det['label_zh']} ({det['score']:.2f})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis_img, label_text, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 在图片顶部添加疾病分类结果
        disease_text = (f"Disease: {result['disease']['name_zh']}"
                       f" ({result['disease']['confidence']:.2%})")
        cv2.putText(vis_img, disease_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 保存可视化图片
        vis_filename = f"{image_path.stem}_pred{image_path.suffix}"
        vis_path = output_dir / vis_filename
        cv2.imencode(image_path.suffix, vis_img)[1].tofile(str(vis_path))
        print(f"Visualization saved: {vis_path}")

        # 保存 LabelMe 兼容 JSON 标注
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
                "imageHeight": orig_h,
                "imageWidth": orig_w,
            }

            json_path = output_dir / f"{image_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)
            print(f"Annotation saved: {json_path}")

        return result

    def predict_directory(
        self,
        image_dir: str,
        output_dir: str = None,
        save_json: bool = True,
    ) -> list[dict]:
        """
        对文件夹中所有图片进行推理。

        返回:
            结果列表
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {image_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_files = sorted(
            f for f in image_dir.iterdir()
            if f.suffix in image_extensions
        )

        if not image_files:
            print(f"No images found in {image_dir}")
            return []

        print(f"Processing {len(image_files)} images from {image_dir}...")
        results = []

        if output_dir is None:
            output_dir = image_dir / "predictions"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img_file in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] {img_file.name}")
            try:
                result = self.predict_and_visualize(
                    str(img_file),
                    output_dir=output_dir,
                    save_json=save_json,
                )
                results.append(result)

                # 打印摘要
                disease = result["disease"]
                print(f"  Disease: {disease['name_zh']} ({disease['confidence']:.2%})")
                print(f"  Detections: {len(result['detections'])} lesions")
                for det in result["detections"]:
                    print(f"    - {det['label_zh']}: "
                          f"bbox={det['bbox']}, score={det['score']:.3f}")

            except Exception as e:
                print(f"  [ERROR] {e}")
                continue

        # 汇总统计
        print(f"\n{'='*50}")
        print(f"SUMMARY: {len(results)}/{len(image_files)} images processed")

        if results:
            disease_counts = {}
            for r in results:
                d = r["disease"]["name_zh"]
                disease_counts[d] = disease_counts.get(d, 0) + 1
            print(f"Disease distribution: {disease_counts}")

            total_detections = sum(len(r["detections"]) for r in results)
            print(f"Total detections: {total_detections}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect and annotate dermatology images"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image file")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to a directory of images")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: outputs/checkpoints/best.pth)")
    parser.add_argument("--backbone", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "convnext_small", "resnet50", "swin_t"])
    parser.add_argument("--no_detection", action="store_true",
                        help="Disable lesion detection (classification only)")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Input image size")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--no_json", action="store_true",
                        help="Skip saving JSON annotations")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be specified")

    # 创建预测器
    predictor = LesionPredictor(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        detection_enabled=not args.no_detection,
        image_size=(args.image_size, args.image_size),
        conf_threshold=args.conf_threshold,
        device=args.device,
    )

    if args.image:
        # 单张图片预测
        result = predictor.predict_and_visualize(
            args.image,
            output_dir=args.output_dir,
            save_json=not args.no_json,
        )

        # 打印详细结果
        print(f"\n{'='*50}")
        print(f"PREDICTION RESULTS: {Path(args.image).name}")
        print(f"{'='*50}")

        disease = result["disease"]
        print(f"\nDisease Classification:")
        print(f"  Top: {disease['name_zh']} ({disease['class']}) "
              f"— {disease['confidence']:.2%}")
        print(f"  All probabilities:")
        for cls_name, prob in sorted(disease["probs"].items(),
                                      key=lambda x: x[1], reverse=True):
            cls_zh = DISEASE_CLASSES_ZH.get(cls_name, cls_name)
            print(f"    {cls_zh} ({cls_name}): {prob:.4f}")

        print(f"\nLesion Detections ({len(result['detections'])} found):")
        if result["detections"]:
            for i, det in enumerate(result["detections"]):
                print(f"  [{i+1}] {det['label_zh']} ({det['label']}) "
                      f"— score: {det['score']:.3f} "
                      f"— bbox: {det['bbox']}")
        else:
            print(f"  (no lesions detected above threshold "
                  f"{args.conf_threshold})")

    elif args.image_dir:
        # 文件夹批量预测
        predictor.predict_directory(
            args.image_dir,
            output_dir=args.output_dir,
            save_json=not args.no_json,
        )


if __name__ == "__main__":
    main()