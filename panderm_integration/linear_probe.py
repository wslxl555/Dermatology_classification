"""
PanDerm 线性探测（linear probe）：冻结 PanDerm ViT，抽 CLS 特征 → 训练线性分类器。

支持两个任务：
  1) --task disease       整图 → 疾病分类（湿疹/类天疱疹/荨麻疹/接触性皮炎，
                          标签来自图片所在目录名，无需 JSON）。
  2) --task lesion_type   病灶 crop → 皮损类型（复用 dermatology_classifier 的
                          lesion_crops 解析 LabelMe，15 类形态）。

权重就绪后会自动加载 PanDerm；没有权重时可用 --dummy 用假编码器验证整条流水线
（无需 timm / PanDerm 权重）。

用法示例:
    # 权重就绪后（真实 linear probe）
    python -m panderm_integration.linear_probe \
        --checkpoint PanDerm/pretrain_weight/panderm_bb_data6_checkpoint-499.pth \
        --variant base --task disease --classes 4

    # 无权重冒烟测试（验证流水线可跑）
    python -m panderm_integration.linear_probe --dummy --task disease --classes 4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from panderm_integration import model as pmodel          # noqa: E402
from panderm_integration.features import (               # noqa: E402
    DummyEncoder, build_eval_transform, load_rgb,
)

# =============================================================================
# 疾病标签（与 derm1m_diagnosis 的 4 病种一致；目录名 → 英文）
# =============================================================================
DISEASE_CN_TO_EN = {
    "接触性皮炎": "contact_dermatitis",
    "湿疹": "eczema",
    "类天疱疹": "bullous_pemphigoid",
    "荨麻疹": "urticaria",
}
# 稳定顺序
DISEASE_CLASSES = ["bullous_pemphigoid", "urticaria", "eczema", "contact_dermatitis"]
DISEASE_TO_IDX = {n: i for i, n in enumerate(DISEASE_CLASSES)}


def extract_patient_id(filename: str) -> str:
    """从文件名提取患者 ID（如 '103_0287.JPG' → '103'）。

    兼容现有工程的约定：前缀数字即患者号；无法识别时用整个 stem。
    """
    stem = Path(filename).stem
    head = stem.split("_")[0].split("-")[0].split(" ")[0]
    return head if head.isdigit() else stem


def parse_args():
    p = argparse.ArgumentParser(description="PanDerm linear probe")
    p.add_argument("--task", choices=["disease", "lesion_type"], default="disease")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="PanDerm 权重路径（省略则自动搜索）")
    p.add_argument("--variant", choices=["base", "large"], default="base")
    p.add_argument("--dummy", action="store_true",
                   help="用假编码器验证流水线（无需权重/timm）")
    p.add_argument("--dummy_dim", type=int, default=768)
    p.add_argument("--classes", type=int, default=4, choices=[3, 4],
                   help="疾病类别数（3=湿疹/类天疱疹/荨麻疹，4=加接触性皮炎）")
    p.add_argument("--labeled_root", type=str, default=str(_PROJECT_ROOT / "labeled"))
    p.add_argument("--unlabeled_root", type=str, default=str(_PROJECT_ROOT / "unlabeled"))
    p.add_argument("--include_unlabeled", action="store_true",
                   help="疾病任务同时纳入 unlabeled/（无标注，但目录名即疾病标签）")
    p.add_argument("--cache_dir", type=str,
                   default=str(_PROJECT_ROOT / "panderm_integration" / "cache"))
    p.add_argument("--norm", choices=["imagenet", "uniform", "clip"], default="imagenet")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--min_size", type=int, default=8, help="病灶 crop 最小尺寸")
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# =============================================================================
# 样本扫描
# =============================================================================
def scan_disease_samples(args):
    """整图 → 疾病分类样本：[(path, disease_idx, patient_id)]。"""
    roots = [Path(args.labeled_root)]
    if args.include_unlabeled:
        roots.append(Path(args.unlabeled_root))
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    samples = []
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            en = DISEASE_CN_TO_EN.get(d.name)
            if en is None:
                continue
            if args.classes == 3 and en == "contact_dermatitis":
                continue
            for img in sorted(d.iterdir()):
                if img.suffix.lower() in exts:
                    samples.append((str(img), DISEASE_TO_IDX[en],
                                    extract_patient_id(img.name)))
    return samples


def scan_lesion_samples(args):
    """病灶 crop → 皮损类型样本：[(crop_array, morph_idx, patient_id)]。"""
    crop_dir = _PROJECT_ROOT / "dermatology_classifier" / "crop_classifier"
    if not crop_dir.exists():
        raise FileNotFoundError(f"未找到 crop 工具目录: {crop_dir}")
    if str(crop_dir) not in sys.path:
        sys.path.insert(0, str(crop_dir))
    import cv2
    import lesion_crops as lc

    labeled = Path(args.labeled_root)
    samples = []
    n_json = 0
    for json_path in sorted(labeled.rglob("*.json")):
        img_path = lc.find_image_for_json(json_path)
        if img_path is None:
            continue
        # 仅保留 4 病种里的
        if img_path.parent.name not in DISEASE_CN_TO_EN:
            continue
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shapes = lc.parse_labelme_shapes(json_path)
        n_json += 1
        for crop, gt_idx in lc.crop_lesions(img, shapes, args.min_size):
            if crop is None:
                continue
            samples.append((crop, int(gt_idx), extract_patient_id(img_path.name)))
    print(f"扫描 {n_json} 个 JSON，得到 {len(samples)} 个病灶 crop")
    return samples


# =============================================================================
# 特征抽取
# =============================================================================
def build_extractor(args):
    """返回 (feature_fn, transform, feat_dim)。feature_fn: PIL 图 → (D,) np 向量。"""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dummy:
        enc = DummyEncoder(dim=args.dummy_dim).to(device).eval()
        dim = args.dummy_dim
    else:
        ckpt = args.checkpoint or pmodel.find_checkpoint(args.variant)
        if ckpt is None:
            raise SystemExit(
                "未找到 PanDerm 权重。请：\n"
                "  1) 下载权重并放到 PanDerm/pretrain_weight/（见 panderm_integration/README.md）\n"
                "  2) 或用 --dummy 验证流水线（无需权重）"
            )
        enc = pmodel.load_panderm(str(ckpt), args.variant, device)
        dim = pmodel.VARIANT_CFG[args.variant]["embed_dim"]

    transform = build_eval_transform(norm=args.norm)

    @torch.no_grad()
    def feature_fn(img):
        from PIL import Image
        if isinstance(img, np.ndarray):          # 病灶 crop 是 numpy RGB
            img = Image.fromarray(img)
        x = transform(img).unsqueeze(0).to(device)
        return enc(x)[0].cpu().numpy().astype(np.float32)

    return feature_fn, transform, dim


def extract_for_samples(feature_fn, transform, samples, args, cache_key):
    """对样本抽特征。样本元素可能是 (path, label, patient) 或 (crop, label, patient)。"""
    cache_path = Path(args.cache_dir) / f"{cache_key}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["features"].astype(np.float32), \
            data["labels"].astype(np.int64), list(data["patients"])

    X, y, patients = [], [], []
    n = len(samples)
    for i, s in enumerate(samples):
        src = s[0]
        if isinstance(src, str):  # 图像路径
            img = load_rgb(src)
            if img is None:
                continue
        else:  # numpy crop (RGB)
            img = src
        X.append(feature_fn(img))
        y.append(s[1])
        patients.append(s[2])
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  特征抽取 {i+1}/{n}", flush=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=X, labels=y, patients=np.array(patients))
    return X, y, patients


# =============================================================================
# 患者级交叉验证评估
# =============================================================================
def patient_level_cv(X, y, patients, args):
    """按患者分组做分层 K-fold，避免同一患者的样本跨 train/test 泄漏。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, p in enumerate(patients):
        groups[p].append(i)
    pat_ids = list(groups.keys())
    # 每个患者取多数标签作为该患者的类别（用于分层）
    pat_label = np.array([Counter(y[groups[p]]).most_common(1)[0][0] for p in pat_ids])

    accs, f1s, aucs = [], [], []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for tr_p, va_p in skf.split(pat_ids, pat_label):
        tr_idx = [i for p in np.array(pat_ids)[tr_p] for i in groups[p]]
        va_idx = [i for p in np.array(pat_ids)[va_p] for i in groups[p]]
        clf = LogisticRegression(max_iter=args.max_iter, class_weight="balanced")
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[va_idx])
        proba = clf.predict_proba(X[va_idx])
        accs.append(accuracy_score(y[va_idx], pred))
        f1s.append(f1_score(y[va_idx], pred, average="macro", zero_division=0))
        try:
            aucs.append(roc_auc_score(y[va_idx], proba, multi_class="ovr",
                                      average="macro"))
        except Exception:
            aucs.append(float("nan"))
    # 过滤掉某折缺少类别导致的 nan AUC
    valid_aucs = [a for a in aucs if np.isfinite(a)]
    auc = (float(np.mean(valid_aucs)), float(np.std(valid_aucs))) if valid_aucs \
        else (float("nan"), float("nan"))
    return {
        "n_patients": len(pat_ids), "n_samples": len(y),
        "acc": (float(np.mean(accs)), float(np.std(accs))),
        "macro_f1": (float(np.mean(f1s)), float(np.std(f1s))),
        "auc": auc,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print(f"PanDerm linear probe — task={args.task} classes={args.classes}")
    print("=" * 60)

    if args.task == "disease":
        samples = scan_disease_samples(args)
        names = [DISEASE_CLASSES[i] for i in range(args.classes)]
        cache_key = f"lp_disease_c{args.classes}_norm{args.norm}_" \
                    f"{'dummy' if args.dummy else args.variant}"
    else:
        samples = scan_lesion_samples(args)
        import lesion_crops as lc
        names = list(lc.MORPHOLOGY_CLASSES)
        cache_key = f"lp_lesion_norm{args.norm}_{'dummy' if args.dummy else args.variant}"

    if not samples:
        raise SystemExit("未扫描到任何样本，请检查数据目录。")

    print(f"样本分布: {Counter(s[1] for s in samples)}")

    feature_fn, transform, dim = build_extractor(args)
    print(f"特征维度: {dim}（{'dummy' if args.dummy else 'PanDerm-' + args.variant}）")

    X, y, patients = extract_for_samples(feature_fn, transform, samples, args, cache_key)
    print(f"抽取特征: X={X.shape}")

    res = patient_level_cv(X, y, patients, args)
    print(f"\n患者级 {args.folds}-fold 交叉验证（LogisticRegression）")
    print(f"患者数 {res['n_patients']} / 样本数 {res['n_samples']} / 类别 {len(names)}")
    print(f"  Accuracy : {res['acc'][0]:.4f} ± {res['acc'][1]:.4f}")
    print(f"  Macro-F1 : {res['macro_f1'][0]:.4f} ± {res['macro_f1'][1]:.4f}")
    print(f"  AUC      : {res['auc'][0]:.4f} ± {res['auc'][1]:.4f}")
    print(f"\n类别: {names}")


if __name__ == "__main__":
    main()
