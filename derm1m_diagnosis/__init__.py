"""
derm1m_diagnosis：Derm1M 对齐的「皮损类型 + 数量 → 皮肤疾病鉴别诊断」离线本体推理包。

用法:
    from derm1m_diagnosis import diagnose, format_report

    result = diagnose({"bulla": 3, "erosion": 1})
    print(format_report(result))
"""

from .diagnose import (
    diagnose,
    normalize_counts,
    format_report,
    diagnose_to_json,
)
from .ontology import (
    DETECTION_CLASSES,
    DETECTION_CLASSES_ZH,
    DISEASE_ONTOLOGY,
    DISEASE_NAMES_ZH,
    DISEASE_ORDER,
)

__all__ = [
    "diagnose",
    "normalize_counts",
    "format_report",
    "diagnose_to_json",
    "DETECTION_CLASSES",
    "DETECTION_CLASSES_ZH",
    "DISEASE_ONTOLOGY",
    "DISEASE_NAMES_ZH",
    "DISEASE_ORDER",
]
