"""
三段式 Derm1M 辅助诊断管线 CLI。

用法：
    # 单张图片
    python -m derm1m_diagnosis.run --image D:/path/img.jpg

    # 整个文件夹
    python -m derm1m_diagnosis.run --image_dir D:/path/folder/

    # 关闭 DermLIP（纯皮损规则模式，无需下载权重）
    python -m derm1m_diagnosis.run --image img.jpg --no_dermlip

    # 只信 DermLIP 视觉信号（alpha=0）或只信规则（alpha=1）
    python -m derm1m_diagnosis.run --image img.jpg --alpha 0.0
    python -m derm1m_diagnosis.run --image img.jpg --alpha 1.0
"""

import argparse
import json
from pathlib import Path

from .pipeline import DiagnosisPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Derm1M 辅助诊断管线")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="单张图片路径")
    g.add_argument("--image_dir", type=str, help="文件夹路径")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="检测模型 checkpoint（默认用 dermatology_detection 的 best.pth）")
    p.add_argument("--image_size", type=int, default=512,
                   help="检测模型输入尺寸（默认 512，与 best.pth 一致）")
    p.add_argument("--conf_threshold", type=float, default=0.3,
                   help="检测框置信度阈值")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="规则(1) vs DermLIP(0) 融合权重")
    p.add_argument("--no_dermlip", action="store_true",
                   help="禁用 DermLIP（无需下载权重，纯规则模式）")
    p.add_argument("--dermlip_model", type=str, default=None,
                   help="DermLIP 模型名（默认 DermLIP_PanDerm-base-w-PubMed-256）")
    p.add_argument("--out", type=str, default=None,
                   help="结果 JSON 输出路径")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def print_result(r: dict):
    print("\n" + "=" * 56)
    print(f"图片: {Path(r['image_path']).name}")
    print("=" * 56)
    print(f"\n{r['diagnosis_text']}")
    print("\n三疾病概率:")
    for d, prob in sorted(r["probs"].items(), key=lambda x: -x[1]):
        zh = r["disease_zh"] if d == r["disease"] else ""
        from .lesion_rules import DISEASE_CLASSES_ZH
        print(f"  {d:22s} {DISEASE_CLASSES_ZH[d]:8s} {prob:.4f}"
              + ("   <-- 综合" if d == r["disease"] else ""))
    if r["signal_dermlip"]:
        print("\n  [信号A 规则]   " + "  ".join(
            f"{d}={v:.3f}" for d, v in r["signal_rule"].items()))
        print("  [信号B DermLIP] " + "  ".join(
            f"{d}={v:.3f}" for d, v in r["signal_dermlip"].items()))
    print(f"\n皮损计数: {r['lesion_counts']}")


def main():
    args = parse_args()
    pipeline = DiagnosisPipeline(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        conf_threshold=args.conf_threshold,
        alpha=args.alpha,
        use_dermlip=not args.no_dermlip,
        dermlip_model=args.dermlip_model,
        device=args.device,
    )

    if pipeline.dermlip is not None and not args.no_dermlip:
        ok = pipeline.dermlip.load()
        print(f"DermLIP 加载: {'成功' if ok else '失败 -> ' + pipeline.dermlip._load_error}")

    results = []
    if args.image:
        results = [pipeline.run(args.image)]
        print_result(results[0])
    else:
        folder = Path(args.image_dir)
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        imgs = sorted(f for f in folder.iterdir() if f.suffix in exts)
        print(f"处理 {len(imgs)} 张图片...")
        for i, f in enumerate(imgs):
            print(f"\n[{i+1}/{len(imgs)}] {f.name}")
            try:
                r = pipeline.run(str(f))
                results.append(r)
                print_result(r)
            except Exception as e:  # noqa: BLE001
                print(f"  [ERROR] {e}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
