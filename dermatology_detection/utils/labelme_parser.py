"""
LabelMe JSON 标注解析器。
从 LabelMe 格式的标注文件中提取边界框和标签。
"""

import json
import re
from pathlib import Path
from typing import Optional


def parse_labelme(json_path: Path) -> Optional[dict]:
    """
    解析 LabelMe JSON 标注文件。

    参数:
        json_path: .json 标注文件的路径。

    返回:
        包含键 'image_path'、'width'、'height'、'annotations' 的字典,
        annotations 是字典列表:{'label': str, 'bbox': [x1,y1,x2,y2]}。
        解析失败时返回 None。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None

    shapes = data.get("shapes", [])
    if not shapes:
        return None

    # 从 JSON 中确定图像路径
    image_path = data.get("imagePath", "")
    if image_path:
        # 尝试相对于 JSON 文件所在位置解析
        img_path = json_path.parent / Path(image_path).name
        if not img_path.exists():
            # 尝试不同的扩展名
            for ext in [".JPG", ".jpg", ".PNG", ".png", ".jpeg", ".JPEG"]:
                alt = img_path.with_suffix(ext)
                if alt.exists():
                    img_path = alt
                    break
    else:
        # 根据 JSON 文件名推断
        stem = json_path.stem
        for ext in [".JPG", ".jpg", ".PNG", ".png"]:
            candidate = json_path.parent / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        else:
            img_path = None

    # 提取标注
    annotations = []
    img_width = data.get("imageWidth", 0)
    img_height = data.get("imageHeight", 0)

    for shape in shapes:
        shape_type = shape.get("shape_type", "")
        if shape_type not in ("rectangle", "polygon"):
            continue

        label = shape.get("label", "").strip()
        if not label:
            continue

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        # 将点转换为 bbox [x1, y1, x2, y2]
        if shape_type == "rectangle":
            (x1, y1), (x2, y2) = points[0], points[1]
        else:
            # 多边形:计算外接边界框
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # 确保坐标顺序正确
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        annotations.append({
            "label": label,
            "bbox": [x1, y1, x2, y2],
        })

    if not annotations:
        return None

    return {
        "image_path": str(img_path) if img_path else None,
        "width": img_width,
        "height": img_height,
        "annotations": annotations,
    }
