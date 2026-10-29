"""
单张图像推理。

用法：
    python -m inference.predict --image path/to/image.jpg --checkpoint best.pth
    python -m inference.predict --image path/to/image.jpg --checkpoint best.pth --visualize
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（支持 python -m inference.predict 从任意目录运行）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch

from data.transforms import get_val_transforms
from models.multi_task_model import build_model
from config import (
    CHECKPOINT_DIR, DISEASE_CLASSES, DISEASE_CLASSES_ZH,
    LESION_CLASSES,
)
from data.label_mapping import LESION_CLASSES as LESION_CLASSES_FULL


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint filename in outputs/checkpoints/")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--visualize", action="store_true",
                        help="Show visualization (requires display)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save annotated image to path (e.g. result.jpg)")
    parser.add_argument("--score_thresh", type=float, default=0.3,
                        help="Score threshold for showing detections")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    image_size: tuple[int, int] = (512, 512),
    device: str = "cuda",
) -> dict:
    """在单张图像上执行推理。

    返回缩放回原始图像坐标的边界框。
    """
    # 路径规范化：将 Windows 风格反斜杠转为正斜杠，兼容 Linux 环境
    image_path = Path(image_path).as_posix()
    if not Path(image_path).exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    # 使用 imdecode 处理包含特殊字符的路径(cv2.imread 对此会失败)
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    transform = get_val_transforms(image_size=image_size)

    # 应用变换,不传 bbox 参数
    transformed = transform(image=img_rgb, bboxes=[], class_labels=[])
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    _, _, new_h, new_w = img_tensor.shape

    # 推理预测
    model.eval()
    with torch.no_grad():
        result = model.predict(img_tensor)

    # 将框从模型输入尺寸缩放回原始图像尺寸
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h
    det_boxes = result["detection_boxes"]
    det_boxes_scaled = []
    for boxes_per_img in det_boxes:
        if len(boxes_per_img) > 0:
            scaled = boxes_per_img.clone()
            scaled[:, 0] *= scale_x  # x1
            scaled[:, 1] *= scale_y  # y1
            scaled[:, 2] *= scale_x  # x2
            scaled[:, 3] *= scale_y  # y2
            det_boxes_scaled.append(scaled)
        else:
            det_boxes_scaled.append(boxes_per_img)

    return {
        "disease_probs": result["disease_probs"].cpu().numpy()[0],
        "disease_pred": result["disease_pred"].cpu().item(),
        "detection_boxes": det_boxes_scaled,
        "detection_labels": result["detection_labels"],
        "detection_scores": result["detection_scores"],
    }


def print_results(result: dict, score_thresh: float = 0.3):
    """美观地打印预测结果。"""
    probs = result["disease_probs"]
    pred = result["disease_pred"]

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    print(f"\nPredicted disease: {DISEASE_CLASSES[pred]} "
          f"({DISEASE_CLASSES_ZH.get(DISEASE_CLASSES[pred], '')})")
    print(f"\nDisease probabilities:")
    for i, cls_name in enumerate(DISEASE_CLASSES):
        zh = DISEASE_CLASSES_ZH.get(cls_name, "")
        bar = "█" * int(probs[i] * 40)
        marker = " ←" if i == pred else ""
        print(f"  {cls_name:25s} ({zh}): {probs[i]:.4f} {bar}{marker}")

    # 检测结果
    boxes = result.get("detection_boxes", [])
    labels = result.get("detection_labels", [])
    scores = result.get("detection_scores", [])

    if boxes and len(boxes) > 0 and len(boxes[0]) > 0:
        b = boxes[0]
        s = scores[0]
        l = labels[0]

        # 按分数阈值过滤
        keep = s > score_thresh
        b = b[keep]
        s = s[keep]
        l = l[keep]

        n = len(b)
        print(f"\nDetected lesions (score > {score_thresh}): {n}")
        for i in range(min(n, 20)):
            box = b[i].cpu().numpy()
            score = s[i].item()
            label_idx = l[i].item()
            if label_idx < len(LESION_CLASSES_FULL):
                label_name = LESION_CLASSES_FULL[label_idx]
            else:
                label_name = f"class_{label_idx}"
            print(f"  [{i}] bbox=({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}) "
                  f"score={score:.3f}  type={label_name}")
    else:
        print("\nNo lesions detected (detection branch may be disabled).")


def visualize_prediction(image_path: str, result: dict, score_thresh: float = 0.3,
                         save_path: str = None):
    """用边界框可视化预测结果。

    参数:
        image_path: 原始图像路径
        result: predict_image 返回的结果字典
        score_thresh: 检测分数阈值
        save_path: 若提供,将结果图保存到该路径而非弹窗显示
    """
    import matplotlib
    if save_path:
        matplotlib.use("Agg")  # 无头模式,不弹窗
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 使用 imdecode 处理包含特殊字符的路径
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)

    # ---- 绘制检测框（皮损 bbox） ----
    boxes = result.get("detection_boxes", [])
    labels = result.get("detection_labels", [])
    scores = result.get("detection_scores", [])

    # 不同皮损类型的颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, len(LESION_CLASSES_FULL)))

    if boxes and len(boxes) > 0 and len(boxes[0]) > 0:
        b = boxes[0].cpu().numpy()
        s = scores[0].cpu().numpy()
        l = labels[0].cpu().numpy()

        for i in range(len(b)):
            if s[i] < score_thresh:
                continue

            x1, y1, x2, y2 = b[i]
            w, h = x2 - x1, y2 - y1
            label_idx = int(l[i])
            color = colors[label_idx % len(colors)]

            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)

            if label_idx < len(LESION_CLASSES_FULL):
                label_name = LESION_CLASSES_FULL[label_idx]
            else:
                label_name = f"class_{label_idx}"

            ax.text(
                x1, y1 - 5,
                f"{label_name} ({s[i]:.2f})",
                color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=color[:3], alpha=0.8),
            )

    # ---- 标题：疾病预测 ----
    probs = result["disease_probs"]
    pred = result["disease_pred"]
    pred_en = DISEASE_CLASSES[pred]
    pred_zh = DISEASE_CLASSES_ZH.get(pred_en, "")

    title = f"Prediction: {pred_en} ({pred_zh})  confidence: {probs[pred]:.2%}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved annotated image to: {save_path}")
    else:
        plt.show()


def main():
    """命令行入口:加载模型并对单张图像执行推理。"""
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"Available checkpoints: {list(CHECKPOINT_DIR.glob('*.pth'))}")
        return

    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = build_model(
        backbone_name=config.get("backbone", "convnext_tiny"),
        detection_enabled=config.get("detection_enabled", False),
        detection_head=config.get("detection_head", "fcos"),
    )
    # 使用 strict=False 以兼容新旧 checkpoint（旧版含已移除的 position_classifier 权重，加载时跳过）
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False,
    )
    if missing:
        print(f"  [INFO] Missing keys (will use random init): {len(missing)} keys")
    if unexpected:
        print(f"  [INFO] Unexpected keys (ignored): {len(unexpected)} keys")
    model = model.to(device)

    # 推理预测
    result = predict_image(
        args.image, model,
        image_size=(args.image_size, args.image_size),
        device=device,
    )

    print_results(result, score_thresh=args.score_thresh)

    if args.visualize or args.save:
        visualize_prediction(args.image, result,
                             score_thresh=args.score_thresh,
                             save_path=args.save)


if __name__ == "__main__":
    main()
