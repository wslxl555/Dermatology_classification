"""
「病灶 crop → 形态 → 疾病」第二段：疾病预测。

对每张有 LabelMe 标注的图，统计其病灶形态直方图（大疱/水疱/丘疹/鳞屑…的比例），
再在【患者级】上用形态直方图训练一个轻量疾病分类器（逻辑回归），
5-fold 交叉验证，回答「病灶形态到底能预测多少疾病信息」。

两种形态来源：
  1) --use-gt（默认）：直接用 LabelMe 里的形态标签（无需训练 crop 分类器，给出上界）
  2) --checkpoint lesion_morph.pth：用第一段训练好的 crop 形态分类器预测形态

用法:
    python predict_by_lesions.py --use-gt
    python predict_by_lesions.py --checkpoint ../outputs/checkpoints/lesion_morph.pth
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

from config import (
    LABELED_ROOT, DISEASE_CN_TO_EN, DISEASE_TO_IDX, DISEASE_CLASSES,
    NUM_DISEASE_CLASSES, DEFAULT_CONFIG, CHECKPOINT_DIR,
)
from data import extract_patient_id
from transforms import get_val_transforms
from model import build_model
from lesion_crops import (
    parse_labelme_shapes, find_image_for_json, crop_lesions,
    MORPHOLOGY_CLASSES,
)

NUM_MORPH = len(MORPHOLOGY_CLASSES)


def parse_args():
    p = argparse.ArgumentParser(description="病灶形态 → 疾病预测")
    p.add_argument("--labeled_root", type=str, default=str(LABELED_ROOT))
    p.add_argument("--use-gt", action="store_true",
                   help="直接用 LabelMe 形态标签（默认；无需 crop 分类器）")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="可选：用训练好的 crop 形态分类器预测形态")
    p.add_argument("--backbone", type=str, default=DEFAULT_CONFIG["backbone"])
    p.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"])
    p.add_argument("--min_size", type=int, default=8)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_crop_classifier(ckpt_path, image_size, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", DEFAULT_CONFIG["backbone"])
    model = build_model(backbone_name=backbone, num_classes=NUM_MORPH,
                        pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tf = get_val_transforms(image_size)
    return model, tf


@torch.no_grad()
def crop_morph_predict(model, tf, device, crop):
    tensor = tf(image=crop)["image"].unsqueeze(0).to(device)
    logits = model(tensor)
    return int(logits.argmax(dim=1).item())


def build_profiles(args):
    """返回 per-image 形态直方图 + 疾病 + 患者。"""
    labeled_root = Path(args.labeled_root)

    crop_model = crop_tf = None
    if args.checkpoint:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        crop_model, crop_tf = load_crop_classifier(args.checkpoint,
                                                   args.image_size, device)
        use_gt = False
    else:
        use_gt = True
        device = None

    profiles = []  # list of dict(disease, patient_id, hist)
    n_img = 0
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
        shapes = parse_labelme_shapes(json_path)
        hist = np.zeros(NUM_MORPH)
        n = 0
        for crop, gt_idx in crop_lesions(image, shapes, args.min_size):
            if use_gt:
                idx = gt_idx
            else:
                idx = crop_morph_predict(crop_model, crop_tf, device, crop)
            hist[idx] += 1
            n += 1
        if n == 0:
            continue
        n_img += 1
        profiles.append({
            "disease": disease,
            "patient_id": extract_patient_id(img_path.name),
            "hist": hist / n,   # 归一化为比例
        })
    return profiles


def patient_level_cv(profiles, n_folds, seed):
    """把 per-image 形态直方图按患者聚合，做患者级 5-fold 交叉验证。"""
    agg = defaultdict(lambda: {"disease": None, "hists": []})
    for p in profiles:
        agg[p["patient_id"]]["disease"] = p["disease"]
        agg[p["patient_id"]]["hists"].append(p["hist"])

    X = np.array([np.mean(v["hists"], axis=0) for v in agg.values()])
    y = np.array([DISEASE_TO_IDX[v["disease"]] for v in agg.values()])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, f1s, aucs = [], [], []
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[va])
        proba = clf.predict_proba(X[va])
        accs.append(accuracy_score(y[va], pred))
        f1s.append(f1_score(y[va], pred, average="macro", zero_division=0))
        try:
            aucs.append(roc_auc_score(y[va], proba, multi_class="ovr",
                                      average="macro"))
        except Exception:
            aucs.append(float("nan"))
    return {
        "n_patients": len(X), "n_images": len(profiles),
        "acc": (np.mean(accs), np.std(accs)),
        "macro_f1": (np.mean(f1s), np.std(f1s)),
        "auc": (np.nanmean(aucs), np.nanstd(aucs)),
    }


def main():
    args = parse_args()
    mode = "GT 形态标签" if (args.checkpoint is None or args.use_gt) else \
           f"crop 分类器 {args.checkpoint}"
    print(f"形态来源: {mode}")

    profiles = build_profiles(args)
    if not profiles:
        print("没有找到任何带标注的图片")
        return

    res = patient_level_cv(profiles, args.folds, args.seed)
    print(f"\n患者级 {args.folds}-fold 交叉验证（形态直方图 → 疾病）")
    print(f"患者数 {res['n_patients']} / 图片数 {res['n_images']}")
    print(f"  Accuracy : {res['acc'][0]:.4f} ± {res['acc'][1]:.4f}")
    print(f"  Macro-F1 : {res['macro_f1'][0]:.4f} ± {res['macro_f1'][1]:.4f}")
    print(f"  AUC      : {res['auc'][0]:.4f} ± {res['auc'][1]:.4f}")
    print(f"\n（对比：整图 CNN 患者级 Accuracy ~0.91，看形态直方图能达到多少）")


if __name__ == "__main__":
    main()
