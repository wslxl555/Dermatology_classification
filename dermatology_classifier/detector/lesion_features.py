"""
病灶特征描述：从检出框（在原图上裁剪病灶区域）提取大小/形状/颜色/纹理特征。

用于可解释性输出 —— 不只说"检出 3 个大疱"，还描述每个病灶的
大小、形状、颜色、纹理，帮助医生/患者理解诊断依据。

所有特征从【框内像素】计算，无需分割 mask；阈值是启发式的（按 0-255 像素值），
仅作粗略描述，后续可用真实物理尺度（如 dermoscopy 的 cm 标尺）替换。
"""

import cv2
import numpy as np


def extract_lesion_features(image_rgb, box):
    """从一张 RGB 原图 + 一个框 (x1,y1,x2,y2) 提取病灶特征。

    返回 dict：width/height/area/rel_area/aspect/roundness/
              mean_rgb/redness/brightness/local_std/lap_var。
    框无效时返回 None。
    """
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_rgb[y1:y2, x1:x2]
    h, w = crop.shape[:2]

    # ---- 大小 ----
    area = float(w * h)
    rel_area = area / float(h_img * w_img)

    # ---- 形状（长宽比 / 圆度）----
    aspect = (w / h) if h > 0 else 0.0
    roundness = (min(w, h) / max(w, h)) if max(w, h) > 0 else 0.0

    # ---- 颜色 ----
    mean_rgb = crop.reshape(-1, 3).mean(axis=0)
    r, g, b = mean_rgb
    redness = float(r - g)                      # 红度（红疹 > 棕褐 > 肤色）
    brightness = float((r + g + b) / 3.0)

    # ---- 纹理 ----
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    local_std = float(gray.std())               # 表面粗糙度（灰度方差）
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # 边界清晰度（梯度能量）

    return {
        "width": w, "height": h, "area": area, "rel_area": rel_area,
        "aspect": aspect, "roundness": roundness,
        "mean_rgb": [float(r), float(g), float(b)],
        "redness": redness, "brightness": brightness,
        "local_std": local_std, "lap_var": lap_var,
    }


def _size_desc(rel_area):
    if rel_area < 0.001:
        return "细小"
    if rel_area < 0.01:
        return "小"
    if rel_area < 0.05:
        return "中等"
    return "大"


def _shape_desc(roundness, aspect):
    if roundness >= 0.85:
        return "圆形"
    if roundness >= 0.6:
        return "椭圆形"
    if aspect >= 2.0 or aspect <= 0.5:
        return "长条形"
    return "不规则"


def _color_desc(redness, brightness):
    # 皮肤科常见：红（炎症）、棕褐（色素）、肤色（不明显）
    if redness > 25:
        return "红色"
    if redness < -5:
        return "青紫"
    if brightness < 90:
        return "深褐"
    if brightness > 170:
        return "肤色"
    return "棕褐"


def _texture_desc(local_std, lap_var):
    parts = []
    parts.append("边界清晰" if lap_var > 500 else
                 ("边界模糊" if lap_var < 120 else "边界中等"))
    parts.append("表面粗糙" if local_std > 45 else
                 ("表面光滑" if local_std < 22 else "表面中等"))
    return "、".join(parts)


def describe_lesion(image_rgb, box):
    """返回病灶的自然语言描述字符串，如
    "中等大小、椭圆形、红色、边界清晰、表面粗糙"。
    """
    f = extract_lesion_features(image_rgb, box)
    if f is None:
        return "（无效框）"
    return (f"{_size_desc(f['rel_area'])}（{_shape_desc(f['roundness'], f['aspect'])}"
            f"、{_color_desc(f['redness'], f['brightness'])}"
            f"、{_texture_desc(f['local_std'], f['lap_var'])}）")


def describe_lesion_full(image_rgb, box):
    """返回病灶特征 dict（含定量值 + 定性描述）。"""
    f = extract_lesion_features(image_rgb, box)
    if f is None:
        return None
    f["size_desc"] = _size_desc(f["rel_area"])
    f["shape_desc"] = _shape_desc(f["roundness"], f["aspect"])
    f["color_desc"] = _color_desc(f["redness"], f["brightness"])
    f["texture_desc"] = _texture_desc(f["local_std"], f["lap_var"])
    f["desc"] = (f"{f['size_desc']}、{f['shape_desc']}、{f['color_desc']}、"
                 f"{f['texture_desc']}")
    return f
