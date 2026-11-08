"""
荨麻疹标签清洗辅助脚本。

扫描「荨麻疹」目录下的图片，按文件名关键词把样本分成亚型，
并标记出「可疑 / 可能混入」的样本，帮助人工复核。

分类规则（按优先级）:
  丘疹性荨麻疹  含 "丘疹性荨麻疹" / "丘疹"
  急性荨麻疹    含 "急性"
  慢性荨麻疹    含 "慢性"
  普通荨麻疹    仅含 "荨麻疹"（无其他特征）
  无荨麻疹关键词 完全不含 "荨麻疹" 等任何已知关键词（多为纯编号文件名，需人工确认）

每个样本额外打两个标记:
  suspicious    含 "?"/"？"/"未诊断"/"待定"/"疑似" 等（诊断不确定）
  other_disease 含 湿疹/皮炎/天疱疹/软疣 等其他疾病关键词（可能放错目录或共病）

「需人工复核」= suspicious 或 other_disease 非空 或 无荨麻疹关键词。

用法:
    python clean_urticaria_labels.py
    python clean_urticaria_labels.py --write-csv outputs/urticaria_review.csv
    python clean_urticaria_labels.py --move outputs/quarantine_urticaria   # 隔离需复核样本
    python clean_urticaria_labels.py --disease 湿疹   # 也可对其他疾病目录跑
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from config import DISEASE_CN_TO_EN, LABELED_ROOT, UNLABELED_ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 可疑标记：诊断不确定
SUSPICIOUS_MARKERS = ["?", "？", "未诊断", "待定", "待诊", "疑似", "可能", "考虑"]

# 其他疾病关键词：出现在荨麻疹目录里可能是放错 / 共病
OTHER_DISEASE = [
    "湿疹", "皮炎", "天疱疮", "类天疱疮", "类天疱疹", "传染性软疣", "软疣",
    "银屑病", "痤疮", "血管炎", "药疹", "扁平疣", "玫瑰糠疹", "脂溢性",
    "神经性皮炎", "接触性", "特应性",
]


def parse_args():
    p = argparse.ArgumentParser(description="荨麻疹标签清洗辅助脚本")
    p.add_argument("--disease", type=str, default="荨麻疹",
                   help="要清洗的疾病目录中文名（默认: 荨麻疹）")
    p.add_argument("--labeled_root", type=str, default=str(LABELED_ROOT))
    p.add_argument("--unlabeled_root", type=str, default=str(UNLABELED_ROOT))
    p.add_argument("--write-csv", type=str, default=None,
                   help="可选：把全部样本分类结果写到 CSV 路径")
    p.add_argument("--move", type=str, default=None,
                   help="可选：把「需复核」样本移动到该目录（隔离，供人工复核）")
    return p.parse_args()


def classify(name: str):
    """返回 (subtype, suspicious, other_list)。"""
    suspicious = any(m in name for m in SUSPICIOUS_MARKERS)
    other = [k for k in OTHER_DISEASE if k in name]
    # 去掉被更长关键词包含的重复项（如 "软疣" ⊂ "传染性软疣"）
    other = sorted(other, key=len, reverse=True)
    dedup = []
    for k in other:
        if not any(k in kept for kept in dedup):
            dedup.append(k)
    other = dedup

    if any(k in name for k in ["丘疹性荨麻疹", "丘疹"]):
        subtype = "丘疹性荨麻疹"
    elif "急性" in name:
        subtype = "急性荨麻疹"
    elif "慢性" in name:
        subtype = "慢性荨麻疹"
    elif "荨麻疹" in name:
        subtype = "普通荨麻疹"
    else:
        subtype = "无荨麻疹关键词"

    return subtype, suspicious, other


def collect_samples(disease_cn, labeled_root, unlabeled_root):
    samples = []
    for root, label in [(labeled_root, "labeled"), (unlabeled_root, "unlabeled")]:
        d = Path(root) / disease_cn
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                samples.append((label, f.name, str(f)))
    return samples


def main():
    args = parse_args()
    disease_cn = args.disease
    if disease_cn not in DISEASE_CN_TO_EN:
        print(f"[警告] 未知疾病目录名: {disease_cn}，"
              f"已知: {list(DISEASE_CN_TO_EN.keys())}")

    samples = collect_samples(disease_cn, args.labeled_root, args.unlabeled_root)
    if not samples:
        print(f"未找到 {disease_cn} 目录下的图片")
        return

    rows = []
    subtype_counter = Counter()
    n_suspicious = n_other = n_nokey = 0
    review = []  # 需人工复核的样本

    for source, name, path in samples:
        subtype, suspicious, other = classify(name)
        subtype_counter[subtype] += 1

        reasons = []
        if suspicious:
            n_suspicious += 1
            reasons.append("诊断不确定(?)")
        if other:
            n_other += 1
            reasons.append("含其他疾病:" + "/".join(other))
        if subtype == "无荨麻疹关键词":
            n_nokey += 1
            reasons.append("无荨麻疹关键词")

        rows.append({
            "source": source, "filename": name, "subtype": subtype,
            "suspicious": suspicious, "other_disease": "/".join(other),
            "review": "; ".join(reasons),
        })
        if reasons:
            review.append((source, name, subtype, "; ".join(reasons), path))

    # ---- 打印统计 ----
    print(f"\n{'='*60}")
    print(f"「{disease_cn}」目录清洗报告（共 {len(samples)} 张）")
    print(f"{'='*60}")
    print("\n亚型分布:")
    for st in ["丘疹性荨麻疹", "急性荨麻疹", "慢性荨麻疹",
               "普通荨麻疹", "无荨麻疹关键词"]:
        if subtype_counter.get(st):
            print(f"  {st:14s}: {subtype_counter[st]} 张")
    print(f"\n标记统计:")
    print(f"  诊断不确定(?)      : {n_suspicious} 张")
    print(f"  含其他疾病关键词    : {n_other} 张")
    print(f"  无荨麻疹关键词      : {n_nokey} 张")
    print(f"  ── 需人工复核合计    : {len(review)} 张")

    # ---- 打印需复核清单 ----
    if review:
        print(f"\n{'='*60}")
        print("需人工复核样本清单:")
        print(f"{'='*60}")
        for source, name, subtype, reason, _path in review:
            print(f"  [{source}] {name}")
            print(f"         -> {subtype} | {reason}")

    # ---- 可选写 CSV ----
    if args.write_csv:
        out = Path(args.write_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "source", "filename", "subtype", "suspicious",
                "other_disease", "review"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV 已保存: {out}")

    # ---- 可选移动（隔离）需复核样本 ----
    if args.move:
        dest = Path(args.move)
        moved = 0
        for source, name, subtype, reason, path in review:
            dst = dest / source / disease_cn / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, str(dst))
            moved += 1
        print(f"\n已移动 {moved} 张需复核样本到: {dest}")


if __name__ == "__main__":
    main()
