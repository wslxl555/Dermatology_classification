"""
Derm1M 对齐的皮损 → 疾病临床本体（离线知识库）。

本模块不依赖 torch / 网络，是纯 Python 数据。它把「皮损类型 + 数量」
映射到 4 个病种（湿疹 / 类天疱疹 / 荨麻疹 / 接触性皮炎）的鉴别诊断证据。

本体依据：
  1. Derm1M 的 130 个临床概念（term_lists/ConceptTerms.json，本仓库已本地存在）。
     概念名与 Derm1M 完全对齐，例如：
       - "bulla"  → bullae / blister
       - "wheal"  → urticaria（Derm1M 直接把 wheal 关联到荨麻疹）
       - "scale"  → flaky and scaly / scaly / hyperkeratosis
       - "vesicle"→ vesicle / fluid-containing
       - "erosion"→ erosive / erosion / breakdown of the outer layers
       - "crust"  → dried crust / crust
       - "lichenification" → thickened and leathery
  2. 标准皮肤病学（Rook's / Fitzpatrick's Dermatology）的皮损-疾病关联。

皮损类别与 dermatology_detection/config.py 的 LESION_CLASSES（15 类）严格一致，
即检测器（inference/detect_lesions.py）的真实输出词汇表。
"""

from pathlib import Path

# =============================================================================
# 皮损检测类别（15 类，与 dermatology_detection/config.py 的 LESION_CLASSES
# 严格一致 —— 检测器 detect_lesions.py 的真实输出）
# =============================================================================
DETECTION_CLASSES = [
    # 原发性皮损
    "macule",          # 斑疹
    "patch",           # 斑片
    "papule",          # 丘疹
    "plaque",          # 斑块
    "nodule",          # 结节
    "vesicle",         # 水疱
    "bulla",           # 大疱（类天疱疹关键信号）
    "pustule",         # 脓疱
    # 继发性皮损
    "scale",           # 鳞屑
    "crust",           # 痂
    "erosion",         # 糜烂
    "excoriation",     # 抓痕
    "scar",            # 瘢痕
    "lichenification", # 苔藓化
    "fissure",         # 皲裂
]

DETECTION_CLASSES_ZH = {
    "macule": "斑疹", "patch": "斑片", "papule": "丘疹", "plaque": "斑块",
    "nodule": "结节", "vesicle": "水疱", "bulla": "大疱", "pustule": "脓疱",
    "scale": "鳞屑", "crust": "痂", "erosion": "糜烂", "excoriation": "抓痕",
    "scar": "瘢痕", "lichenification": "苔藓化", "fissure": "皲裂",
    # 兼容 dermatology_classifier/detector 的旧 10 类词汇（"other" 为低频合并桶）
    "other": "其他",
}

DETECTION_CLASS_ZH_TO_EN = {zh: en for en, zh in DETECTION_CLASSES_ZH.items()}

# =============================================================================
# Derm1M 临床概念 → 检测类别 的映射
# =============================================================================
# Derm1M 的 130 个概念里，与「原发性/继发性皮损形态」相关的概念映射到
# 本项目的 15 类检测类别。两个特殊映射：
#   * "wheal"(风团) 是荨麻疹的特征性损害，但本项目检测器没有独立的 wheal 类，
#     临床/检测上最接近「丘疹(papule)/斑块(plaque)」，故映射到二者。
#   * "erythema"(红斑) 是颜色概念，最接近「斑疹(macule)」（红斑性斑疹）。
DERM1M_CONCEPT_TO_DETECTION = {
    "macule": "macule",
    "patch": "patch",
    "papule": "papule",
    "plaque": "plaque",
    "nodule": "nodule",
    "vesicle": "vesicle",
    "bulla": "bulla",
    "pustule": "pustule",
    "scale": "scale",
    "crust": "crust",
    "erosion": "erosion",
    "excoriation": "excoriation",
    "scar": "scar",
    "lichenification": "lichenification",
    "fissure": "fissure",
    "wheal": "plaque",           # 风团 → 斑块（荨麻疹特征性损害）
    "erythema": "macule",        # 红斑 → 斑疹（红斑性斑疹）
    "ulcer": "erosion",          # 溃疡 → 糜烂（本项目无独立溃疡类）
    "atrophy": "scar",           # 萎缩 → 瘢痕（近似）
    "xerosis": "scale",          # 干燥 → 鳞屑（近似，表面干燥脱屑）
    "induration": "nodule",      # 硬结/水肿 → 结节（近似）
}

# 反向映射：检测类别 -> 可能的 Derm1M 概念（用于把检测结果翻译回 Derm1M 术语）
DETECTION_TO_DERM1M_CONCEPTS = {}
for _concept, _det in DERM1M_CONCEPT_TO_DETECTION.items():
    DETECTION_TO_DERM1M_CONCEPTS.setdefault(_det, []).append(_concept)

# =============================================================================
# 疾病本体：4 个病种的皮损证据
# =============================================================================
# 每种病种定义三类证据（值 = 诊断权重，可正可负）：
#   key        关键/特征性损害（权重高，检出即强烈支持）
#   supportive 支持性损害（常见但不特异）
#   against    反对性损害（该损害对该病不典型，检出即降低其可能）
#
# 权重为启发式、临床可解释；正 = 支持，负 = 反对。计数通过饱和函数参与
# （见 diagnose.py），使「3 个大疱」比「1 个大疱」证据更强，但封顶不线性放大。
DISEASE_ONTOLOGY = {
    "bullous_pemphigoid": {
        "name_zh": "类天疱疹",
        # 紧张性大疱：类天疱疹的核心/特征性信号
        "key": {"bulla": 3.0},
        # 大疱破裂后形成糜烂、结痂；可伴水疱、红斑、荨麻疹样斑块
        "supportive": {
            "erosion": 1.5, "crust": 1.0, "vesicle": 1.0,
            "macule": 0.5, "plaque": 0.5,
        },
        "against": {},
        "note": "紧张性大疱(bulla)是类天疱疹的核心信号，大疱破裂后形成糜烂、结痂。",
    },
    "urticaria": {
        "name_zh": "荨麻疹",
        # 风团(wheal)：荨麻疹特征性损害。Derm1M 概念 'wheal' 直接对应 urticaria。
        # 本项目检测器无 wheal 类，风团在检测中表现为丘疹/斑块。
        "key": {"plaque": 1.5, "papule": 1.5},
        "supportive": {"macule": 0.5},
        # 风团为短暂性、非鳞屑、非结痂的隆起；鳞屑/结痂/大疱/糜烂/苔藓化都反对荨麻疹
        "against": {
            "scale": -2.0, "crust": -1.5, "bulla": -1.5, "erosion": -1.0,
            "lichenification": -1.0, "fissure": -1.0, "excoriation": -0.5,
        },
        "note": "风团(wheal)是荨麻疹特征性损害（Derm1M 概念 'wheal' 直接关联 urticaria）；"
                "风团通常无鳞屑、无结痂、无糜烂、无苔藓化。",
    },
    "eczema": {
        "name_zh": "湿疹",
        # 湿疹无单一特征性损害，靠「丘疹 + 水疱 + 鳞屑 + 苔藓化/抓痕」的组合
        "key": {},
        "supportive": {
            "scale": 1.5, "vesicle": 1.5, "lichenification": 1.2,
            "papule": 1.0, "erosion": 1.0, "crust": 1.0, "excoriation": 1.0,
            "macule": 0.5,
        },
        # 大疱对湿疹不典型（更倾向类天疱疹/接触性皮炎）
        "against": {"bulla": -0.5},
        "note": "湿疹为多形性损害组合（丘疹+水疱+鳞屑+苔藓化/抓痕），无单一特征性损害。",
    },
    "contact_dermatitis": {
        "name_zh": "接触性皮炎",
        # 急性：红斑、水疱、大疱、渗出（糜烂/结痂）；慢性：鳞屑、苔藓化、皲裂
        "key": {},
        "supportive": {
            "vesicle": 1.5, "erosion": 1.5, "bulla": 1.5, "crust": 1.0,
            "scale": 1.0, "lichenification": 1.0, "fissure": 0.8, "macule": 0.5,
        },
        "against": {},
        "note": "接触性皮炎急性期为红斑、水疱、大疱、渗出（糜烂/结痂），慢性期为鳞屑、苔藓化、皲裂。",
    },
}

# 病种顺序（用于稳定排序输出）
DISEASE_ORDER = [
    "bullous_pemphigoid",
    "urticaria",
    "eczema",
    "contact_dermatitis",
]

# 病种中文名
DISEASE_NAMES_ZH = {d: v["name_zh"] for d, v in DISEASE_ONTOLOGY.items()}

# Derm1M ConceptTerms.json 的本地路径（用于 grounding / 同义词解析）
DERM1M_CONCEPT_TERMS_PATH = (
    Path(__file__).resolve().parent.parent
    / "Derm1M" / "concept_annotation" / "term_lists" / "ConceptTerms.json"
)

# Derm1M 概念同义词 → 概念名 的惰性缓存（读取 ConceptTerms.json 构建）
_SYNONYM_CACHE: dict[str, str] | None = None


def _load_synonyms() -> dict[str, str]:
    """从 Derm1M ConceptTerms.json 构建 {同义词: 概念名} 映射（含概念名本身）。

    失败（文件不存在 / JSON 损坏）时返回空映射，不影响核心推理。
    """
    global _SYNONYM_CACHE
    if _SYNONYM_CACHE is not None:
        return _SYNONYM_CACHE
    mapping: dict[str, str] = {}
    try:
        import json
        if DERM1M_CONCEPT_TERMS_PATH.exists():
            data = json.loads(DERM1M_CONCEPT_TERMS_PATH.read_text(encoding="utf-8"))
            for concept, terms in data.items():
                c = concept.strip().lower()
                mapping[c] = c
                for t in terms:
                    mapping[t.strip().lower()] = c
    except Exception:
        pass
    _SYNONYM_CACHE = mapping
    return mapping


def iter_all_lesions() -> list[str]:
    """返回本体中所有被引用的检测类别（去重、按检测类别顺序）。"""
    seen = set()
    for disease in DISEASE_ONTOLOGY.values():
        for role in ("key", "supportive", "against"):
            for lesion in disease.get(role, {}):
                seen.add(lesion)
    return [c for c in DETECTION_CLASSES if c in seen]


def coerce_lesion_name(name: str) -> str | None:
    """把任意皮损名（英文检测类 / Derm1M 概念 / 中文 / 同义词）归一化到检测类别。

    返回 None 表示无法识别。
    """
    if not isinstance(name, str):
        return None
    n = name.strip().lower()
    if not n:
        return None

    # 直接是检测类别（含 15 类 + 旧 "other"）
    if n in DETECTION_CLASSES or n == "other":
        return n
    # 中文检测类名
    if n in DETECTION_CLASS_ZH_TO_EN:
        return DETECTION_CLASS_ZH_TO_EN[n]
    # Derm1M 概念名（含复合词如 "brown(hyperpigmentation)"，忽略）
    if n in DERM1M_CONCEPT_TO_DETECTION:
        return DERM1M_CONCEPT_TO_DETECTION[n]
    # Derm1M 概念同义词（如 "blister"→bulla、"scaly"→scale、"bullae"→bulla）
    concept = _load_synonyms().get(n)
    if concept is not None and concept in DERM1M_CONCEPT_TO_DETECTION:
        return DERM1M_CONCEPT_TO_DETECTION[concept]
    return None


# 供外部 import 的便捷别名
LESION_CLASSES = DETECTION_CLASSES
LESION_CLASSES_ZH = DETECTION_CLASSES_ZH
NUM_LESION_CLASSES = len(DETECTION_CLASSES)
