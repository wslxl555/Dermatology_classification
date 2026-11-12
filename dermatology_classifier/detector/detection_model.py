"""
病灶检测核心模块：类别定义、标签映射、可切换检测头工厂、数据集。

检测头可自由切换（torchvision 内置，接口统一）：
  - faster_rcnn   Faster R-CNN + FPN
  - retinanet     RetinaNet（anchor-based，类 FCOS）
  - fcos          FCOS（anchor-free）

用途：为整图疾病分类提供可解释性 —— 检出"哪些病灶、几个、在哪"，
解释分类结果（如"预测为类天疱疹，因检出 3 个大疱"）。

注意：为保留诊断关键信号，这里【没有】把 bulla（大疱）并入 other；
低频但无诊断价值的类别（脓疱/抓痕/皲裂/苔藓化/瘢痕/斑丘疹）才并入 other。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# 让本模块能 import 到上层 dermatology_classifier 及 crop_classifier
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "crop_classifier"))

from data import extract_patient_id  # noqa: E402
from lesion_crops import (  # noqa: E402
    parse_labelme_shapes, find_image_for_json,
    resolve_morphology_label, MORPHOLOGY_CLASSES,
)

# =============================================================================
# 检测类别（10 类 = 9 个诊断相关类 + other）
# =============================================================================
DETECTION_CLASSES = [
    "macule",     # 斑疹
    "patch",      # 斑片
    "papule",     # 丘疹
    "plaque",     # 斑块
    "vesicle",    # 水疱
    "scale",      # 鳞屑
    "crust",      # 痂
    "erosion",    # 糜烂
    "bulla",      # 大疱（类天疱疹关键信号，保留）
    "other",      # 其他（脓疱/抓痕/皲裂/苔藓化/瘢痕/斑丘疹）
]

DETECTION_CLASSES_ZH = {
    "macule": "斑疹", "patch": "斑片", "papule": "丘疹", "plaque": "斑块",
    "vesicle": "水疱", "scale": "鳞屑", "crust": "痂", "erosion": "糜烂",
    "bulla": "大疱", "other": "其他",
}

NUM_DETECTION_CLASSES = len(DETECTION_CLASSES)          # 前景类 10
NUM_MODEL_CLASSES = NUM_DETECTION_CLASSES + 1           # 含背景 = 11

DETECTION_HEAD_NAMES = ["faster_rcnn", "retinanet", "fcos"]

# 15 类形态 → 10 类检测（低频并入 other）
_MORPH_TO_DET = {
    "papule": "papule", "plaque": "plaque", "patch": "patch", "macule": "macule",
    "maculopapule": "other", "vesicle": "vesicle", "bulla": "bulla",
    "pustule": "other", "scale": "scale", "crust": "crust", "erosion": "erosion",
    "excoriation": "other", "fissure": "other", "lichenification": "other",
    "scar": "other",
}


def map_label_to_class(raw_label: str) -> int:
    """把 LabelMe 原始标签映射到检测类别（0..9，前景，不含背景）。"""
    idx15 = resolve_morphology_label(raw_label)
    if idx15 is None:
        return DETECTION_CLASSES.index("other")
    det_name = _MORPH_TO_DET.get(MORPHOLOGY_CLASSES[idx15], "other")
    return DETECTION_CLASSES.index(det_name)


# =============================================================================
# 可切换检测头工厂
# =============================================================================
def build_detector(head: str, num_classes: int = NUM_MODEL_CLASSES,
                   offline: bool = False):
    """构建 torchvision 检测模型。num_classes 含背景（= 前景数 + 1）。

    返回模型后，label 约定为：背景 = 0，前景 = 1..N。
    """
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2, fcos_resnet50_fpn,
    )
    from torchvision.models import ResNet50_Weights

    head = head.lower()
    backbone_weights = ResNet50_Weights.IMAGENET1K_V1

    if head in ("faster_rcnn", "fasterrcnn", "faster-rcnn"):
        return fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=backbone_weights, num_classes=num_classes)
    if head in ("retinanet", "retina_net", "retina-net"):
        return retinanet_resnet50_fpn_v2(
            weights=None, weights_backbone=backbone_weights, num_classes=num_classes)
    if head in ("fcos",):
        return fcos_resnet50_fpn(
            weights=None, weights_backbone=backbone_weights, num_classes=num_classes)
    raise ValueError(f"Unknown detection head: {head} "
                     f"(可选: {DETECTION_HEAD_NAMES})")


# =============================================================================
# 数据扫描：labeled/ 下的图片 + LabelMe 标注 → 检测样本
# =============================================================================
def scan_detection_samples(labeled_root):
    """扫描 labeled/，返回检测样本列表。

    每个样本: image_path / json_path / boxes[N,4] / labels[N]（0..9 前景）/
              disease / patient_id。
    跳过没有框或无法解析的图片。
    """
    labeled_root = Path(labeled_root)
    samples = []
    for json_path in sorted(labeled_root.rglob("*.json")):
        img_path = find_image_for_json(json_path)
        if img_path is None:
            continue
        shapes = parse_labelme_shapes(json_path)
        boxes, labels = [], []
        for label, (x1, y1, x2, y2) in shapes:
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(map_label_to_class(label))
        if not boxes:
            continue
        disease = img_path.parent.name  # 疾病中文目录名
        samples.append({
            "image_path": str(img_path),
            "json_path": str(json_path),
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "disease": disease,
            "patient_id": extract_patient_id(img_path.name),
        })
    return samples


# =============================================================================
# 数据集
# =============================================================================
class LesionDetectionDataset(Dataset):
    """病灶检测数据集。返回 (image_tensor, target)。"""

    def __init__(self, samples, target_size=(512, 512), flip=False):
        self.samples = samples
        self.target_size = target_size
        self.flip = flip

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = cv2.imdecode(np.fromfile(s["image_path"], dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = s["boxes"].copy()
        labels = s["labels"].copy()

        image, boxes = self._resize_and_flip(image, boxes)
        # [0,1] 张量；torchvision 检测模型内部会做 ImageNet 归一化
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            # torchvision 检测模型约定背景=0，前景 1..N
            "labels": torch.as_tensor(labels + 1, dtype=torch.int64),
        }
        return image, target

    def _resize_and_flip(self, image, boxes):
        h, w = image.shape[:2]
        th, tw = self.target_size
        image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA
                           if th < h else cv2.INTER_LINEAR)
        sx, sy = tw / w, th / h
        scale = np.array([sx, sy, sx, sy], dtype=np.float32)
        boxes = boxes * scale
        if self.flip and np.random.rand() < 0.5:
            image = image[:, ::-1].copy()
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = tw - x2
            boxes[:, 2] = tw - x1
        return image, boxes


def collate_fn(batch):
    """检测用 collate：不 stack 图片，返回 (images, targets) 两个 list。"""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


# =============================================================================
# 分割式检测头（attention_unet / mamba / sam_adapter / transunet / transcascade）
# =============================================================================
# 这些是「分割式检测」头（bbox 渲染成 mask 弱监督分割，推理时连通域转回框），
# 已在 dermatology_detection 工程实现并可通过 train_labeled.py --detection_head 切换。
#
# 由于两个工程都有顶层 config.py / models/ 等模块，跨进程 import 会发生模块名冲突，
# 因此这里【不】直接 import，改用子进程桥接（见 detector/train_segmentation.py）。
SEGMENTATION_HEAD_NAMES = [
    "attention_unet", "mamba", "sam_adapter", "transunet", "transcascade",
]
