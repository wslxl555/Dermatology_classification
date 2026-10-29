"""
标签标准化:将 217+ 个原始标注标签(双语、含拼写错误)
映射到 15 个标准化的皮损类别。

原始标签来自 4 个疾病文件夹下的 LabelMe JSON 文件:
  - 湿疹: 英文标签 (Plaque, Papule, Patch, ...)
  - 类天疱疮: 英文标签 (Papule+Vesicle, Bulla, ...)
  - 接触性皮炎: 中文标签 (斑片, 丘疹, ...)
  - 荨麻疹: 中文标签 (丘疹, 斑片+X痂, ...)

需要处理的质量问题:
  1. 拼写错误 "Pacth" → "Patch"
  2. 小写 "patch" → "Patch"
  3. 零宽空格字符 (U+200B)
  4. 空格不一致 ("Papule+ Scale" vs "Papule+Scale")
  5. 分隔符不一致(中文中的 + vs / vs ,)
"""

import re

# =============================================================================
# 标准化皮损类别(9 类 = 8 个高频类 + "other")
# =============================================================================
# 低频类别 nodule / bulla / pustule / excoriation / scar / lichenification /
# fissure 在标注中样本极少,单独建模会拖累检测 mAP,统一合并为 "other"。
LESION_CLASSES = [
    "macule",          # 斑疹
    "patch",           # 斑片
    "papule",          # 丘疹
    "plaque",          # 斑块
    "vesicle",         # 水疱
    "scale",           # 鳞屑
    "crust",           # 痂
    "erosion",         # 糜烂
    "other",           # 其他(结节/大疱/脓疱/抓痕/瘢痕/苔藓化/皲裂)
]

LESION_CLASS_TO_IDX = {name: idx for idx, name in enumerate(LESION_CLASSES)}

# =============================================================================
# 中文 → 英文基础映射
# =============================================================================
ZH_TO_EN_BASE = {
    # 原发性皮损(高频,保留)
    "斑疹": "macule",
    "斑片": "patch",
    "丘疹": "papule",
    "斑块": "plaque",
    "水疱": "vesicle",
    # 继发性皮损(高频,保留)
    "鳞屑": "scale",
    "痂": "crust",
    "糜烂": "erosion",
    # 低频皮损 → 合并为 other
    "结节": "other",
    "大疱": "other",
    "脓疱": "other",
    "抓痕": "other",
    "瘢痕": "other",
    "苔藓化": "other",
    "皲裂": "other",
    # 复合 / 特殊
    "斑丘疹": "maculopapule",  # 将被拆分为 [macule, papule]
    "红斑": "macule",          # 红斑 = 红斑性斑疹
    "渗出": "erosion",         # 渗出 = 渗出液,最接近糜烂
    "色沉": "other",           # 炎症后色素沉着 ≈ 瘢痕(低频)
    "血痂": "crust",           # 血痂 = 出血性痂
}

# 拆分为多个基础类别的复合词
ZH_COMPOUND_SPLIT = {
    "斑丘疹": ["macule", "papule"],
    "红斑+丘疹": ["macule", "papule"],
}

# =============================================================================
# 原始标签清洗
# =============================================================================

def clean_label(raw_label: str) -> str:
    """
    清洗并标准化原始标签字符串。
    处理:零宽字符、空格不一致、已知拼写错误、大小写。
    """
    # 移除零宽字符
    cleaned = raw_label.replace("​", "").replace("‌", "").replace("‍", "")
    cleaned = cleaned.replace("﻿", "").replace(" ", " ")

    # 规范化空白字符
    cleaned = " ".join(cleaned.split())

    # 修复已知拼写错误
    cleaned = cleaned.replace("Pacth", "Patch")  # "Pacth" → "Patch"
    cleaned = re.sub(r'\bpatch\b', 'Patch', cleaned)  # 小写 patch

    # 规范化 + 和 / 周围的空格
    cleaned = re.sub(r'\s*\+\s*', '+', cleaned)
    cleaned = re.sub(r'\s*/\s*', '/', cleaned)
    cleaned = re.sub(r'\s*,\s*', '+', cleaned)  # 将逗号视为 +

    # 移除首尾的 +
    cleaned = cleaned.strip("+ ")

    return cleaned


# =============================================================================
# 英文标签 → 标准化类别集合
# =============================================================================

# 将每个英文基础词映射到其标准化类别
# 低频类别(Nodule/Bulla/Pustule/Excoriation/Scar/Lichenification/Fissure)
# 统一映射到 "other",缓解长尾分布对检测的影响。
EN_BASE_TO_CLASS = {
    "Macule": "macule",
    "Maculopapule": "macule",  # 将通过多标签与 papule 一起出现
    "Patch": "patch",
    "Papule": "papule",
    "Plaque": "plaque",
    "Vesicle": "vesicle",
    "Scale": "scale",
    "Crust": "crust",
    "Erosion": "erosion",
    "Nodule": "other",
    "Bulla": "other",
    "Pustule": "other",
    "Excoriation": "other",
    "Scar": "other",
    "Lichenification": "other",
    "Fissure": "other",
}


def parse_english_label(label: str) -> list[str]:
    """
    将英文标签(可能包含 + 或 / 组合)
    解析为标准化的皮损类别名称列表。

    示例:
        "Plaque+Scale" → ["plaque", "scale"]
        "Patch/Vesicle" → ["patch", "vesicle"]
        "Pacth+Bulla+Erosion+Crust" → ["patch", "bulla", "erosion", "crust"]
    """
    cleaned = clean_label(label)

    # 按 + 或 / 拆分
    parts = re.split(r'[+/]', cleaned)

    classes = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        cls = EN_BASE_TO_CLASS.get(part)
        if cls:
            classes.append(cls)

    return list(set(classes))  # 去重


def parse_chinese_label(label: str) -> list[str]:
    """
    将中文标签解析为标准化的皮损类别名称。

    示例:
        "斑片+丘疹" → ["patch", "papule"]
        "丘疹/斑疹+痂" → ["papule", "macule", "crust"]
        "斑丘疹+鳞屑" → ["macule", "papule", "scale"]
        "痂" → ["crust"]
    """
    cleaned = clean_label(label)

    # 先处理复合词(斑丘疹、红斑+丘疹 等)
    for compound, base_list in ZH_COMPOUND_SPLIT.items():
        if compound in cleaned:
            # 用其英文展开并用 + 连接后替换复合词
            cleaned = cleaned.replace(compound, "+".join(base_list))

    # 按分隔符拆分
    # 中文标签使用 +、/、、、(、)、  +
    parts = re.split(r'[+/、（）()]', cleaned)

    classes = []
    for part in parts:
        part = part.strip(" ()（）")
        if not part:
            continue

        # 尝试直接中文 → 英文映射
        en = ZH_TO_EN_BASE.get(part)
        if en is None:
            # 该部分可能已经是英文类别名(来自复合词拆分)
            if part in LESION_CLASS_TO_IDX:
                classes.append(part)
            continue

        if en in LESION_CLASS_TO_IDX:
            classes.append(en)
        elif en == "maculopapule":
            classes.extend(["macule", "papule"])

    return list(set(classes))


# =============================================================================
# 统一标签解析
# =============================================================================

def parse_label(raw_label: str) -> list[str]:
    """
    将任意原始标签(中文或英文)解析为标准化的
    皮损类别名称列表。

    参数:
        raw_label: 来自 LabelMe JSON 的原始标签字符串。

    返回:
        标准化皮损类别名称列表(例如 ["plaque", "scale"])。
        如果标签无法解析,则返回空列表。
    """
    cleaned = clean_label(raw_label)
    if not cleaned:
        return []

    # 检测标签是否主要为中文
    has_chinese = bool(re.search(r'[一-鿿]', cleaned))

    if has_chinese:
        return parse_chinese_label(cleaned)
    else:
        return parse_english_label(cleaned)


def label_to_multi_hot(raw_labels: list[str]) -> list[int]:
    """
    将原始标签列表转换为长度为 NUM_LESION_CLASSES 的 multi-hot 向量。

    每个原始标签都会被解析;如果是组合标签,则其所有组成类别
    都被置为 1。

    参数:
        raw_labels: 一张图像标注中的原始标签字符串列表。

    返回:
        长度为 NUM_LESION_CLASSES 的 multi-hot 列表。
    """
    multi_hot = [0] * len(LESION_CLASSES)
    all_classes = set()

    for raw in raw_labels:
        parsed = parse_label(raw)
        all_classes.update(parsed)

    for cls_name in all_classes:
        if cls_name in LESION_CLASS_TO_IDX:
            multi_hot[LESION_CLASS_TO_IDX[cls_name]] = 1

    return multi_hot


def label_to_indices(raw_labels: list[str]) -> list[int]:
    """
    将原始标签转换为整数类别索引列表。
    用于检测/分类训练。

    返回图像中出现的去重后的类别索引列表。
    """
    multi_hot = label_to_multi_hot(raw_labels)
    return [i for i, v in enumerate(multi_hot) if v]


# =============================================================================
# 校验 / 发现
# =============================================================================

def discover_all_labels(data_root: str) -> set[str]:
    """
    遍历整个数据集,发现所有唯一的原始标签。
    用于构建/更新映射表。
    """
    import json
    import os

    all_labels = set()

    for root, dirs, files in os.walk(data_root):
        for f in files:
            if not f.endswith(".json"):
                continue
            filepath = os.path.join(root, f)
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                if size_mb > 5:
                    continue  # 跳过大文件
                with open(filepath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for shape in data.get("shapes", []):
                    label = shape.get("label", "").strip()
                    if label:
                        all_labels.add(label)
            except Exception:
                pass

    return all_labels


def diagnose_unmapped_labels(data_root: str) -> dict[str, int]:
    """
    查找无法映射到任何标准化类别的标签。
    返回 {原始标签: 数量} 的 dict。
    """
    import json
    import os
    from collections import Counter

    unmapped = Counter()

    for root, dirs, files in os.walk(data_root):
        for f in files:
            if not f.endswith(".json"):
                continue
            filepath = os.path.join(root, f)
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                if size_mb > 5:
                    continue
                with open(filepath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for shape in data.get("shapes", []):
                    raw = shape.get("label", "").strip()
                    if not raw:
                        continue
                    parsed = parse_label(raw)
                    if not parsed:
                        unmapped[raw] += 1
            except Exception:
                pass

    return dict(unmapped.most_common())
