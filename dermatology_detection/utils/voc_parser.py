"""
Pascal VOC XML 标注解析器。
从 VOC 格式的 XML 文件中提取边界框和标签。
用于类天疱疮(bullous pemphigoid)数据集。
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def parse_voc_xml(xml_path: Path) -> Optional[dict]:
    """
    解析 Pascal VOC 格式的 XML 标注文件。

    参数:
        xml_path: .xml 标注文件的路径。

    返回:
        包含键 'image_path'、'width'、'height'、'annotations' 的字典。
        解析失败时返回 None。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, OSError):
        return None

    # 获取图像尺寸
    size = root.find("size")
    if size is not None:
        width = int(size.find("width").text) if size.find("width") is not None else 0
        height = int(size.find("height").text) if size.find("height") is not None else 0
    else:
        width, height = 0, 0

    # 获取图像文件名
    filename = root.find("filename")
    if filename is not None and filename.text:
        img_name = filename.text
        # 尝试在 XML 文件旁查找图像
        img_path = xml_path.parent / img_name
        if not img_path.exists():
            # 尝试其他扩展名
            for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg"]:
                alt = img_path.with_suffix(ext)
                if alt.exists():
                    img_path = alt
                    break
    else:
        img_path = None

    # 提取对象(边界框)
    annotations = []
    for obj in root.findall("object"):
        name = obj.find("name")
        if name is None:
            continue
        label = name.text.strip()
        if not label:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (AttributeError, ValueError):
            continue

        # 确保 bbox 有效
        if xmax <= xmin or ymax <= ymin:
            continue

        annotations.append({
            "label": label,
            "bbox": [xmin, ymin, xmax, ymax],
        })

    if not annotations:
        return None

    return {
        "image_path": str(img_path) if img_path else None,
        "width": width,
        "height": height,
        "annotations": annotations,
    }
