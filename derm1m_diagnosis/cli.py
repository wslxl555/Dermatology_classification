"""
命令行入口：皮损检测结果 → Derm1M 本体鉴别诊断。

支持三种输入方式（互斥）：
  1) --detection-json FILE   消费 dermatology_detection/inference/detect_lesions.py
                             的 --out 输出（{filename: [{bbox,label,score}, ...]}）。
  2) --image FILE --detector-json FILE
                             直接读一张图对应的检测 JSON（单个文件的列表）。
  3) --counts "bulla:3,erosion:1"
                             直接给定皮损计数，用于快速验证 / 离线演示。

用法示例:
    python -m derm1m_diagnosis.cli --counts "bulla:3,erosion:1,crust:1"
    python -m derm1m_diagnosis.cli --counts "papule:2,plaque:1,macule:1"
    python -m derm1m_diagnosis.cli --detection-json det.json
"""

import argparse
import json
import sys
from pathlib import Path

from .diagnose import diagnose, format_report, diagnose_to_json
from .ontology import coerce_lesion_name, DETECTION_CLASSES_ZH


def parse_counts(spec: str) -> dict[str, int]:
    """解析 "bulla:3,erosion:1" 或 "bulla,erosion" 形式的计数。"""
    counts: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, n = part.rsplit(":", 1)
            name, n = name.strip(), n.strip()
        else:
            name, n = part.strip(), "1"
        if not name:
            continue
        cls = coerce_lesion_name(name)
        if cls is None:
            print(f"警告: 忽略无法识别的皮损名 '{name}'", file=sys.stderr)
            continue
        counts[cls] = counts.get(cls, 0) + int(n)
    return counts


def _load_detection_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"检测 JSON 不存在: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _counts_from_detections(detections: list) -> dict[str, int]:
    """从 [{bbox,label,score}, ...] 提取皮损计数。"""
    counts: dict[str, int] = {}
    for d in detections:
        label = d.get("label") if isinstance(d, dict) else None
        cls = coerce_lesion_name(label)
        if cls is None:
            continue
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Derm1M 本体鉴别诊断")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--counts", type=str,
                   help="皮损计数，如 'bulla:3,erosion:1'")
    g.add_argument("--detection-json", type=str,
                   help="detect_lesions.py 的 --out 输出（{文件名: 检测列表}）")
    g.add_argument("--image", type=str, help="图片路径（配合 --detector-json）")
    ap.add_argument("--detector-json", type=str,
                    help="单张图片对应的检测列表 JSON（配合 --image）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非可读报告")
    args = ap.parse_args(argv)

    if args.counts:
        counts = parse_counts(args.counts)
    elif args.detection_json:
        data = _load_detection_json(args.detection_json)
        # {文件名: [检测列表]} —— 对每张图分别诊断
        if args.json:
            out = {}
            for fname, dets in data.items():
                out[fname] = diagnose(_counts_from_detections(dets))
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return
        for fname, dets in data.items():
            print(f"\n【{fname}】")
            print(format_report(diagnose(_counts_from_detections(dets))))
        return
    elif args.image and args.detector_json:
        dets = _load_detection_json(args.detector_json)
        counts = _counts_from_detections(dets)
        print(f"图片: {args.image}")
        if args.json:
            print(diagnose_to_json(diagnose(counts)))
            return
        print(format_report(diagnose(counts)))
        return
    else:
        ap.error("--image 需搭配 --detector-json 使用")

    if args.json:
        print(diagnose_to_json(diagnose(counts)))
    else:
        print(format_report(diagnose(counts)))


if __name__ == "__main__":
    main()
