"""
纯疾病分类推理：只输入图片，输出疾病类别与概率。

完全不读取 .json 标注文件 —— 只需要一张图片即可完成疾病三分类。

用法:
    python -m inference.predict_disease --image D:/test.jpg
    python -m inference.predict_disease --image_dir D:/test_images/
    python -m inference.predict_disease --image test.jpg --checkpoint best.pth --image_size 320
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import DISEASE_CLASSES, DISEASE_CLASSES_ZH, CHECKPOINT_DIR
from models.classification_model import build_classification_model
from data.transforms import get_cls_val_transforms


class DiseasePredictor:
    """纯疾病分类预测器。"""

    def __init__(
        self,
        checkpoint_path: str = None,
        backbone: str = "convnext_tiny",
        image_size: int = 224,
        device: str = None,
    ):
        self.image_size = (image_size, image_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "disease_cls_best.pth"
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Train first with: python train_disease_classifier.py"
            )

        self.model = build_classification_model(backbone_name=backbone, pretrained=False)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_cls_val_transforms(self.image_size)

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"  epoch: {ckpt.get('epoch', 'unknown')} | best val F1: {ckpt.get('best_val_f1', 'N/A')}")

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        对单张图片做疾病分类。只使用图像本身。

        返回:
            {"class": str, "name_zh": str, "confidence": float, "probs": dict}
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR,
        )
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        x = self.transform(image=img_rgb)["image"].unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())

        return {
            "class": DISEASE_CLASSES[pred],
            "name_zh": DISEASE_CLASSES_ZH.get(DISEASE_CLASSES[pred], ""),
            "confidence": float(probs[pred]),
            "probs": {DISEASE_CLASSES[i]: float(probs[i])
                      for i in range(len(DISEASE_CLASSES))},
        }

    def predict_directory(self, image_dir: str) -> list[dict]:
        """对文件夹内所有图片做分类。"""
        image_dir = Path(image_dir)
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_files = sorted(f for f in image_dir.iterdir() if f.suffix in exts)
        if not image_files:
            print(f"No images found in {image_dir}")
            return []

        results = []
        for i, img_file in enumerate(image_files):
            try:
                r = self.predict(str(img_file))
                results.append({"image": img_file.name, **r})
                print(f"[{i+1}/{len(image_files)}] {img_file.name}: "
                      f"{r['name_zh']} ({r['confidence']:.2%})")
            except Exception as e:
                print(f"[{i+1}/{len(image_files)}] {img_file.name}: [ERROR] {e}")

        return results


def main():
    p = argparse.ArgumentParser(description="Pure disease classification inference (image-only)")
    p.add_argument("--image", type=str, default=None, help="Single image path")
    p.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path (default: outputs/checkpoints/disease_cls_best.pth)")
    p.add_argument("--backbone", type=str, default="convnext_tiny",
                   choices=["convnext_tiny", "convnext_small", "resnet50", "swin_t"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if not args.image and not args.image_dir:
        p.error("Either --image or --image_dir must be specified")

    predictor = DiseasePredictor(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        image_size=args.image_size,
        device=args.device,
    )

    if args.image:
        r = predictor.predict(args.image)
        print(f"\nImage: {args.image}")
        print(f"  Disease: {r['name_zh']} ({r['class']}) — {r['confidence']:.2%}")
        print("  Probabilities:")
        for cls_name, prob in sorted(r["probs"].items(), key=lambda x: -x[1]):
            print(f"    {DISEASE_CLASSES_ZH.get(cls_name, cls_name)} ({cls_name}): {prob:.4f}")

    elif args.image_dir:
        predictor.predict_directory(args.image_dir)


if __name__ == "__main__":
    main()
