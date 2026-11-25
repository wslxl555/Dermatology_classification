"""
Stage2/3 的符号规则层：皮损类型 + 数量 → 疾病证据打分。

这层对应 Derm1M 数据集的 ontology.json 概念层——把检测到的皮损类型
（恰好是 Derm1M 的临床概念）映射为对三种疾病的诊断证据权重。

证据权重基于临床皮肤病学常识手工设定，可解释、可标定：
  - 类天疱疹(bullous_pemphigoid)：大疱(bulla)是决定性信号，糜烂(erosion)强信号。
  - 湿疹(eczema)：丘疹/斑块 + 鳞屑/痂/苔藓化 + 抓痕，是"炎症+角质异常+瘙痒"的组合。
  - 荨麻疹(urticaria)：无鳞屑/痂/糜烂/瘢痕（无表皮破坏），以斑片/斑块(风团)为主。

注意：当前 15 类皮损里没有 wheal(风团)——荨麻疹的标志性皮损，
因此荨麻疹在本层只能靠"缺乏鳞屑/糜烂"的负证据 + DermLIP 视觉信号兜底。
"""

import math

# 与 dermatology_detection/config.py 的 LESION_CLASSES 保持一致（15 类）
LESION_CLASSES = [
    "macule", "patch", "papule", "plaque", "nodule",
    "vesicle", "bulla", "pustule", "scale", "crust",
    "erosion", "excoriation", "scar", "lichenification", "fissure",
]

# 与 dermatology_detection/config.py 的 DISEASE_CLASSES 保持一致（3 类）
DISEASE_CLASSES = ["eczema", "bullous_pemphigoid", "urticaria"]

DISEASE_CLASSES_ZH = {
    "eczema": "湿疹",
    "bullous_pemphigoid": "大疱型类天疱疹",
    "urticaria": "荨麻疹",
}

LESION_CLASSES_ZH = {
    "macule": "斑疹",
    "patch": "斑片",
    "papule": "丘疹",
    "plaque": "斑块",
    "nodule": "结节",
    "vesicle": "水疱",
    "bulla": "大疱",
    "pustule": "脓疱",
    "scale": "鳞屑",
    "crust": "痂",
    "erosion": "糜烂",
    "excoriation": "抓痕",
    "scar": "瘢痕",
    "lichenification": "苔藓化",
    "fissure": "皲裂",
}

# 证据权重：disease -> {lesion_type: weight}
# weight > 0 为正向证据；负数为反向证据（出现该皮损反而降低该疾病可能性）。
EVIDENCE_WEIGHTS = {
    "eczema": {
        "macule": 0.3,          # 红斑
        "patch": 0.3,
        "papule": 0.6,          # 丘疹是急性湿疹常见原发皮损
        "plaque": 0.5,
        "nodule": 0.1,          # 结节性痒疹与慢性湿疹相关
        "vesicle": 0.4,         # 急性湿疹可有小水疱
        "bulla": -0.3,          # 大疱不典型
        "pustule": 0.1,         # 继发感染偶见
        "scale": 0.7,           # 鳞屑（苔藓化/干燥）
        "crust": 0.6,           # 痂（抓破后）
        "erosion": 0.5,         # 糜烂（渗出）
        "excoriation": 0.3,     # 抓痕——剧烈瘙痒
        "scar": 0.2,            # 慢性湿疹可遗留瘢痕/色沉
        "lichenification": 0.6, # 苔藓化——慢性湿疹标志
        "fissure": 0.5,         # 皲裂——慢性湿疹（如手部）
    },
    "bullous_pemphigoid": {
        "macule": 0.2,          # 前驱期可有荨麻疹样红斑
        "patch": 0.2,
        "papule": 0.0,
        "plaque": 0.2,
        "nodule": 0.0,
        "vesicle": 0.5,         # 早期小水疱
        "bulla": 1.0,           # 大疱——关键信号
        "pustule": 0.1,         # 偶见无菌性脓疱
        "scale": -0.2,          # 鳞屑不典型
        "crust": 0.5,           # 疱破后结痂
        "erosion": 0.8,         # 大疱破溃后糜烂——强信号
        "excoriation": 0.2,     # 瘙痒抓痕
        "scar": 0.1,
        "lichenification": 0.0,
        "fissure": 0.0,
    },
    "urticaria": {
        "macule": 0.3,          # 风团初起表现为红斑
        "patch": 0.4,           # 大片风团
        "papule": 0.2,
        "plaque": 0.4,          # 荨麻疹样斑块
        "nodule": 0.0,
        "vesicle": -0.3,        # 无疱
        "bulla": -0.4,          # 无大疱
        "pustule": 0.0,
        "scale": -0.5,          # 无鳞屑——关键负证据
        "crust": -0.3,          # 无痂
        "erosion": -0.4,        # 无表皮破坏
        "excoriation": 0.2,     # 荨麻疹亦剧烈瘙痒
        "scar": -0.3,           # 荨麻疹消退不留痕
        "lichenification": 0.0,
        "fissure": -0.3,        # 无皲裂
    },
}


def count_lesions(detections: list[dict]) -> dict[str, int]:
    """从检测结果统计每种皮损数量。

    detections 为 LesionPredictor.predict() 返回的 detection 列表，
    每项含 "label" 字段（英文，如 "bulla"）。未知类别忽略。
    """
    counts = {k: 0 for k in LESION_CLASSES}
    for det in detections:
        label = det.get("label", "")
        if label in counts:
            counts[label] += 1
    return counts


def score_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """皮损计数 → 三种疾病的证据 logits。

    对数量取 log1p 以抑制单类型超大量级的支配效应。
    """
    scores = {}
    for disease in DISEASE_CLASSES:
        s = 0.0
        for lesion, weight in EVIDENCE_WEIGHTS[disease].items():
            c = counts.get(lesion, 0)
            s += weight * math.log1p(c)
        scores[disease] = s
    return scores


def explain_from_counts(counts: dict[str, int]) -> str:
    """把皮损计数转成一段中文临床描述。"""
    present = [
        f"{LESION_CLASSES_ZH.get(l, l)}×{c}"
        for l, c in counts.items() if c > 0
    ]
    if not present:
        return "未检测到明确皮损"
    return "皮损谱：" + "、".join(present)
