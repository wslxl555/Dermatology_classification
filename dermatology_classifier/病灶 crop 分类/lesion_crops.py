"""
病灶 crop 相关核心工具。

职责：
  - 解析 LabelMe 标注 JSON（得到每个病灶的 label + 锚框 bbox）
  - 从原图裁剪病灶 crop
  - 把（可能复合的）形态 label 解析成统一形态类别
  - 构建病灶 crop 数据集（供形态分类器训练用）

LabelMe 的 shape.label 形如 "Papule（丘疹）+Vesicle（水疱）" 或
"Vesicle（水疱）/Bulla（大疱）"（斜杠 = 不确定，二者都算）。
用正则提取英文形态词，映射到 MORPHOLOGY_CLASSES；复合标签按诊断优先级解析为单标签。
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# 让本目录内的脚本能 import 到上层 dermatology_classifier 的 config/data/...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LABELED_ROOT  # noqa: E402
from data import extract_patient_id  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 形态类别（单标签，用于 crop 分类）
MORPHOLOGY_CLASSES = [
    "papule",           # 丘疹
    "plaque",           # 斑块
    "patch",            # 斑片
    "macule",           # 斑疹
    "maculopapule",     # 斑丘疹
    "vesicle",          # 水疱
    "bulla",            # 大疱
    "pustule",          # 脓疱
    "scale",            # 鳞屑
    "crust",            # 痂
    "erosion",          # 糜烂
    "excoriation",      # 抓痕
    "fissure",          # 皲裂
    "lichenification",  # 苔藓化
    "scar",             # 瘢痕
]
MORPH_IDX = {name: i for i, name in enumerate(MORPHOLOGY_CLASSES)}

# 复合标签解析为单标签时的优先级（诊断价值从高到低；数字越小越优先）
MORPH_PRIORITY = [
    "bulla", "pustule", "vesicle", "lichenification", "scale",
    "erosion", "fissure", "excoriation", "crust", "maculopapule",
    "papule", "plaque", "patch", "macule", "scar",
]
_PRIORITY_ORDER = {name: i for i, name in enumerate(MORPH_PRIORITY)}


def parse_morphology_set(label: str):
    """从 label 字符串提取形态类别索引集合。"""
    words = re.findall(r"[A-Za-z]+", label)
    idxs = set()
    for w in words:
        wl = w.lower()
        if wl in MORPH_IDX:
            idxs.add(MORPH_IDX[wl])
    return idxs


def resolve_morphology_label(label: str):
    """把（可能复合的）label 解析为单个形态类别索引；无法识别返回 None。"""
    idxs = parse_morphology_set(label)
    if not idxs:
        return None
    return min(idxs, key=lambda i: _PRIORITY_ORDER[MORPHOLOGY_CLASSES[i]])


def parse_labelme_shapes(json_path):
    """读取 LabelMe JSON，返回 [(label, bbox)]，bbox=(x1,y1,x2,y2)。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    shapes = []
    for s in data.get("shapes", []):
        if s.get("shape_type", "rectangle") != "rectangle":
            continue
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        shapes.append((s["label"], bbox))
    return shapes


def find_image_for_json(json_path):
    """根据 JSON 文件名找到同 stem 的图片路径（大小写不敏感）。"""
    stem = json_path.stem
    for f in json_path.parent.iterdir():
        if f.is_file() and f.stem == stem and f.suffix.lower() in IMAGE_EXTS:
            return f
    return None


def crop_lesions(image, shapes, min_size=8):
    """从原图按 shapes 裁剪病灶，返回 [(crop_rgb, morph_idx)]，跳过过小/无法识别的。"""
    h, w = image.shape[:2]
    out = []
    for label, (x1, y1, x2, y2) in shapes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue
        idx = resolve_morphology_label(label)
        if idx is None:
            continue
        out.append((image[y1:y2, x1:x2], idx))
    return out


def build_crop_dataset(labeled_root=None, min_size=8):
    """扫描 labeled/ 下所有 .json，裁剪病灶。

    返回 (crops, labels, metas)：
      crops  : list[np.ndarray]  RGB 病灶图
      labels : list[int]         形态类别索引
      metas  : list[(img_path, patient_id)] 来源图片与患者
    """
    labeled_root = Path(labeled_root or LABELED_ROOT)
    crops, labels, metas = [], [], []
    n_json = 0
    for json_path in sorted(labeled_root.rglob("*.json")):
        img_path = find_image_for_json(json_path)
        if img_path is None:
            continue
        n_json += 1
        image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shapes = parse_labelme_shapes(json_path)
        pid = extract_patient_id(img_path.name)
        for crop, idx in crop_lesions(image, shapes, min_size):
            crops.append(crop)
            labels.append(idx)
            metas.append((str(img_path), pid))
    return crops, labels, metas
