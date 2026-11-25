"""
Derm1M 本体推理引擎：皮损类型 + 数量 → 皮肤疾病鉴别诊断。

纯 Python（无 torch / 无网络依赖）。输入一张图的皮损计数（或检测框列表），
输出按「支持度」降序排列的鉴别诊断，附每个病种的证据（支持/反对/缺失关键损害）。

设计要点：
  * 计数参与：单一皮损的诊断贡献按 `min(count, cap)` 饱和（默认 cap=3），
    使「3 个大疱」比「1 个大疱」更强，但 20 个不会线性放大 20 倍。
  * 支持度是「启发式证据分」，不是校准概率；只在病种间做相对排序。
  * 无任何支持证据（所有病种 score<=0）时返回「证据不足」。
"""

from __future__ import annotations

import json
import math
from collections import Counter

from .ontology import (
    DETECTION_CLASSES,
    DETECTION_CLASSES_ZH,
    DISEASE_ONTOLOGY,
    DISEASE_ORDER,
    DISEASE_NAMES_ZH,
    coerce_lesion_name,
)


def _saturate(count: int, cap: int = 3) -> float:
    """计数饱和：min(count, cap)。"""
    return float(min(max(count, 0), cap))


def normalize_counts(lesions, cap: int = 3) -> dict[str, int]:
    """把任意皮损输入归一化为 {检测类别: 计数}。

    lesions 支持：
      * dict / Counter：{皮损名: 数量}
      * list[str]：皮损名列表（可含重复，自动计数），
        例如 ["bulla", "bulla", "erosion"] 或 [{"label": "bulla"}, ...]
    """
    counts: Counter[str] = Counter()
    if lesions is None:
        return {}
    if isinstance(lesions, dict):
        for name, n in lesions.items():
            cls = coerce_lesion_name(name)
            if cls is not None:
                counts[cls] += int(n)
        return dict(counts)
    if isinstance(lesions, (list, tuple)):
        for item in lesions:
            if isinstance(item, dict):
                name = item.get("label") or item.get("class") or item.get("name")
            else:
                name = item
            cls = coerce_lesion_name(name)
            if cls is not None:
                counts[cls] += 1
        return dict(counts)
    raise TypeError(f"无法解析的皮损输入类型: {type(lesions)}")


def _score_disease(counts: dict[str, int], profile: dict, cap: int) -> dict:
    """对单个病种计算支持度与证据明细。"""
    score = 0.0
    matched: list[dict] = []
    against: list[dict] = []
    missing_key: list[str] = []

    # 权重已带符号（key/supportive 为正，against 为负），直接累加。
    for role in ("key", "supportive", "against"):
        for lesion, weight in profile.get(role, {}).items():
            n = counts.get(lesion, 0)
            if n > 0:
                contrib = weight * _saturate(n, cap)
                score += contrib
                entry = {
                    "lesion": lesion,
                    "lesion_zh": DETECTION_CLASSES_ZH.get(lesion, lesion),
                    "count": n,
                    "role": role,
                    "weight": weight,
                    "contrib": round(contrib, 3),
                }
                if role == "against":
                    against.append(entry)
                else:
                    matched.append(entry)

    # 关键损害缺失
    for lesion in profile.get("key", {}):
        if counts.get(lesion, 0) == 0:
            missing_key.append(lesion)

    matched.sort(key=lambda e: -e["contrib"])
    against.sort(key=lambda e: e["contrib"])  # 负的在前

    return {
        "disease": None,  # 由调用方填入
        "name_zh": profile["name_zh"],
        "score": round(score, 3),
        "matched": matched,
        "against": against,
        "missing_key": missing_key,
        "note": profile.get("note", ""),
    }


def diagnose(lesions, cap: int = 3, temperature: float = 1.0) -> dict:
    """对一张图的皮损计数做 Derm1M 本体鉴别诊断。

    参数:
        lesions: dict / list（见 normalize_counts）。
        cap: 计数饱和上限。
        temperature: softmax 温度（越小越尖锐；仅影响 confidence 相对值）。

    返回:
        {
          "counts": {检测类别: 计数},          # 归一化后的皮损计数
          "top_disease": str | None,           # 最高支持度病种（无则 None）
          "rankings": [ {disease, name_zh, score, confidence, matched,
                         against, missing_key, note}, ... ],  # 按 score 降序
          "insufficient": bool,                # 是否证据不足
        }
    """
    counts = normalize_counts(lesions, cap)

    results = []
    for disease in DISEASE_ORDER:
        profile = DISEASE_ONTOLOGY[disease]
        r = _score_disease(counts, profile, cap)
        r["disease"] = disease
        results.append(r)

    # 按支持度降序
    results.sort(key=lambda r: -r["score"])

    # softmax 归一化为「相对置信度」（仅用于展示病种间的相对可能性）
    scores = [r["score"] for r in results]
    max_s = max(scores) if scores else 0.0
    if temperature <= 0:
        temperature = 1.0
    exps = [math.exp((s - max_s) / temperature) for s in scores]
    denom = sum(exps) or 1.0
    for r, e in zip(results, exps):
        r["confidence"] = round(e / denom, 4)

    insufficient = max_s <= 0.0
    top = None if insufficient else results[0]["disease"]

    return {
        "counts": {k: v for k, v in sorted(counts.items())},
        "top_disease": top,
        "top_name_zh": DISEASE_NAMES_ZH.get(top, "") if top else None,
        "rankings": results,
        "insufficient": insufficient,
    }


# =============================================================================
# 可读化输出
# =============================================================================
def _fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "（未检出皮损）"
    parts = []
    # 先按检测类别顺序，再补上任何非标准键（如旧词汇 "other"）
    ordered = list(DETECTION_CLASSES) + [k for k in counts if k not in DETECTION_CLASSES]
    for cls in ordered:
        if counts.get(cls):
            zh = DETECTION_CLASSES_ZH.get(cls, cls)
            parts.append(f"{zh}({cls})×{counts[cls]}")
    return "、".join(parts)


def format_report(result: dict) -> str:
    """把 diagnose() 的结果格式化为可读的中文报告。"""
    lines = []
    lines.append("=" * 60)
    lines.append("皮损统计")
    lines.append("=" * 60)
    lines.append(f"检出皮损: {_fmt_counts(result['counts'])}")

    if result["insufficient"]:
        lines.append("\n诊断: 证据不足 —— 未检出足够的关键/支持性皮损，"
                     "建议结合病史与皮肤镜复核。")
        return "\n".join(lines)

    lines.append("")
    lines.append("=" * 60)
    lines.append("Derm1M 本体鉴别诊断（按支持度降序）")
    lines.append("=" * 60)

    for rank, r in enumerate(result["rankings"], 1):
        lines.append(
            f"\n{rank}. {r['name_zh']} ({r['disease']})  "
            f"支持度 {r['score']:+.2f}  相对置信度 {r['confidence']:.1%}"
        )
        if r["matched"]:
            ev = "；".join(
                f"{e['lesion_zh']}×{e['count']}"
                f"({'关键' if e['role']=='key' else '支持'} +{e['contrib']:+.2f})"
                for e in r["matched"]
            )
            lines.append(f"   支持证据: {ev}")
        if r["against"]:
            ev = "；".join(
                f"{e['lesion_zh']}×{e['count']} (反对 {e['contrib']:+.2f})"
                for e in r["against"]
            )
            lines.append(f"   反对证据: {ev}")
        if r["missing_key"]:
            miss = "、".join(DETECTION_CLASSES_ZH.get(l, l)
                             for l in r["missing_key"])
            lines.append(f"   缺失关键损害: {miss}")
        if r["note"]:
            lines.append(f"   说明: {r['note']}")

    lines.append("")
    lines.append("-" * 60)
    top = result["rankings"][0]
    lines.append(
        f"结论: 最符合「{top['name_zh']}」"
        f"（支持度 {top['score']:+.2f}）"
    )
    return "\n".join(lines)


def diagnose_to_json(result: dict) -> str:
    """序列化为 JSON（供程序间传递，与 detect_lesions.py 的输出可拼接）。"""
    return json.dumps(result, ensure_ascii=False, indent=2)
