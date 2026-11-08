"""
晚融合对照实验：CNN 特征 + 病灶形态直方图。

在【同一患者级】5-fold 划分上，用逻辑回归对比三种输入，看形态是否在视觉特征
之外提供互补信息：

  1) CNN 特征        —— 冻结 ImageNet backbone 提的视觉特征
  2) 病灶形态直方图   —— GT 形态标签的比例（上界）
  3) CNN + 形态(晚融合) —— 把两者拼接

关键设计：
  - 特征提取用【冻结的 ImageNet backbone】，不参与训练 → 任何划分都无数据泄漏。
  - 三种模型都用逻辑回归 + StandardScaler，保证公平对比。
  - 患者级聚合：同一患者多图的特征/直方图取均值，再按患者做 5-fold CV。

注意：这里用的是冻结 backbone（非你微调后的 swin_t），所以 CNN 特征那行是
「通用视觉特征」的下界；但结论（形态是否互补）对微调模型同样成立。

用法:
    python late_fusion.py --backbone swin_t --image_size 224 --offline
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import (
    LABELED_ROOT, DISEASE_CN_TO_EN, DISEASE_TO_IDX,
    NUM_DISEASE_CLASSES, DEFAULT_CONFIG,
)
from data import extract_patient_id
from transforms import get_val_transforms
from model import build_model, BACKBONE_NAMES
from lesion_crops import (
    parse_labelme_shapes, find_image_for_json, crop_lesions, MORPHOLOGY_CLASSES,
)

NUM_MORPH = len(MORPHOLOGY_CLASSES)


def parse_args():
    p = argparse.ArgumentParser(description="晚融合对照实验（CNN 特征 + 形态）")
    p.add_argument("--labeled_root", type=str, default=str(LABELED_ROOT))
    p.add_argument("--backbone", type=str, default=DEFAULT_CONFIG["backbone"],
                   choices=BACKBONE_NAMES)
    p.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"])
    p.add_argument("--min_size", type=int, default=8)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--offline", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def extract_cnn_feature(model, image, tf, device):
    tensor = tf(image=image)["image"].unsqueeze(0).to(device)
    feats = model.backbone(tensor)
    feats = torch.flatten(feats, 1)
    return feats[0].cpu().numpy()


def build_records(args, model, tf, device):
    """为每张有标注的图，提取 CNN 特征 + 形态直方图。"""
    labeled_root = Path(args.labeled_root)
    records = []
    for json_path in sorted(labeled_root.rglob("*.json")):
        img_path = find_image_for_json(json_path)
        if img_path is None:
            continue
        disease = DISEASE_CN_TO_EN.get(img_path.parent.name)
        if disease is None:
            continue
        image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cnn_feat = extract_cnn_feature(model, image, tf, device)
        shapes = parse_labelme_shapes(json_path)
        hist = np.zeros(NUM_MORPH)
        n = 0
        for _, gt_idx in crop_lesions(image, shapes, args.min_size):
            hist[gt_idx] += 1
            n += 1
        if n == 0:
            continue
        records.append({
            "disease": disease,
            "patient_id": extract_patient_id(img_path.name),
            "cnn_feat": cnn_feat,
            "hist": hist / n,
        })
    return records


def aggregate_by_patient(records):
    agg = defaultdict(lambda: {"disease": None, "cnn": [], "hist": []})
    for r in records:
        a = agg[r["patient_id"]]
        a["disease"] = r["disease"]
        a["cnn"].append(r["cnn_feat"])
        a["hist"].append(r["hist"])
    X_cnn = np.array([np.mean(v["cnn"], axis=0) for v in agg.values()])
    X_morph = np.array([np.mean(v["hist"], axis=0) for v in agg.values()])
    y = np.array([DISEASE_TO_IDX[v["disease"]] for v in agg.values()])
    return X_cnn, X_morph, y


def run_cv(X_cnn, X_morph, y, n_folds, seed):
    models = {
        "CNN 特征": X_cnn,
        "形态直方图": X_morph,
        "CNN+形态(晚融合)": np.hstack([X_cnn, X_morph]),
    }
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_model = {name: {"acc": [], "f1": [], "auc": []} for name in models}
    for tr, va in skf.split(X_cnn, y):
        for name, X in models.items():
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr])
            Xva = scaler.transform(X[va])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xva)
            proba = clf.predict_proba(Xva)
            per_model[name]["acc"].append(accuracy_score(y[va], pred))
            per_model[name]["f1"].append(
                f1_score(y[va], pred, average="macro", zero_division=0))
            try:
                per_model[name]["auc"].append(
                    roc_auc_score(y[va], proba, multi_class="ovr", average="macro"))
            except Exception:
                per_model[name]["auc"].append(float("nan"))
    return per_model


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(backbone_name=args.backbone, num_classes=NUM_DISEASE_CLASSES,
                        pretrained=True, offline=args.offline).to(device)
    model.eval()
    tf = get_val_transforms(args.image_size)

    print(f"提取特征（冻结 {args.backbone} @ {args.image_size}px）...")
    records = build_records(args, model, tf, device)
    X_cnn, X_morph, y = aggregate_by_patient(records)
    print(f"共 {len(records)} 张标注图，{len(y)} 名患者，"
          f"CNN 特征维数 {X_cnn.shape[1]}，形态维数 {X_morph.shape[1]}")

    per_model = run_cv(X_cnn, X_morph, y, args.folds, args.seed)

    def ms(xs):
        xs = np.asarray(xs, dtype=float)
        return xs.mean(), xs.std()

    print(f"\n患者级 {args.folds}-fold 交叉验证（逻辑回归，mean ± std）")
    print(f"{'模型':22s} {'Accuracy':>20s} {'Macro-F1':>20s} {'AUC':>20s}")
    for name, m in per_model.items():
        a, f, u = ms(m["acc"]), ms(m["f1"]), ms(m["auc"])
        print(f"{name:22s} {a[0]:.4f}±{a[1]:.3f}   {f[0]:.4f}±{f[1]:.3f}   "
              f"{u[0]:.4f}±{u[1]:.3f}")


if __name__ == "__main__":
    main()
