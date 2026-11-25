"""
三段式诊断管线编排：

    image
     ├─ Stage1 皮损检测 + 类型   复用 dermatology_detection 的 LesionPredictor
     ├─ Stage2 皮损类型计数       lesion_rules.count_lesions
     └─ Stage3 疾病诊断决策
          信号A 符号规则(ontology 概念层)   lesion_rules.score_from_counts
          信号B DermLIP 视觉零样本(整图+病灶crop)  dermlip.DermLIPClient
         → 融合(softmax 后加权) → 最终诊断 + 可解释文字

融合：两信号各自 softmax 成概率后按 alpha 加权；DermLIP 不可用时退化为纯规则。
"""

import sys
from pathlib import Path

import numpy as np

from .lesion_rules import (
    DISEASE_CLASSES, DISEASE_CLASSES_ZH,
    count_lesions, score_from_counts, explain_from_counts,
)
from .dermlip import DermLIPClient

# dermatology_detection 仓库根，用于复用其检测模型
_DET_ROOT = Path(__file__).resolve().parent.parent / "dermatology_detection"

# 疾病诊断描述语料库（Signal C 检索用；英文，与 DermLIP 训练语料一致）
DISEASE_DESCRIPTIONS = {
    "eczema": "Eczematous dermatitis with erythematous papules, plaques, "
              "scaling and excoriation, often with crust and lichenification.",
    "bullous_pemphigoid": "Tense bullae and vesicles on erythematous skin, "
                          "with erosions and crusted lesions, consistent with "
                          "bullous pemphigoid.",
    "urticaria": "Evanescent wheals and urticarial plaques without scaling or "
                 "epidermal disruption, consistent with urticaria.",
}


def softmax(logits: dict[str, float]) -> dict[str, float]:
    names = list(logits.keys())
    arr = np.array([logits[n] for n in names], dtype=float)
    arr = arr - arr.max()
    e = np.exp(arr)
    p = e / e.sum()
    return {n: float(v) for n, v in zip(names, p)}


def _crop_to_pil(img_bgr, bbox, pad: int = 8):
    """从 BGR 图像按 bbox 裁剪并转 PIL（RGB）。"""
    from PIL import Image
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - pad); y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad); y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    crop = crop[:, :, ::-1]  # BGR -> RGB
    return Image.fromarray(np.ascontiguousarray(crop))


class DiagnosisPipeline:
    def __init__(
        self,
        checkpoint_path: str = None,
        image_size: int = 512,
        conf_threshold: float = 0.3,
        alpha: float = 0.5,
        use_dermlip: bool = True,
        dermlip_model: str = None,
        device: str = None,
    ):
        """
        alpha: 信号A(规则) vs 信号B(DermLIP) 的融合权重。
               alpha=1.0 表示只用规则，alpha=0.0 表示只用 DermLIP。
        """
        self.conf_threshold = conf_threshold
        self.alpha = alpha
        self.use_dermlip = use_dermlip
        self.device = device or "cpu"

        self.predictor = None
        self._load_predictor(checkpoint_path, image_size)

        self.dermlip = None
        if use_dermlip:
            self.dermlip = DermLIPClient(
                model_name=dermlip_model or DermLIPClient.DEFAULT_MODEL,
                device=self.device,
            )

    def _load_predictor(self, checkpoint_path, image_size):
        if str(_DET_ROOT) not in sys.path:
            sys.path.insert(0, str(_DET_ROOT))
        try:
            from inference.predict_labeled import LesionPredictor
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"无法导入 dermatology_detection 的 LesionPredictor：{e}\n"
                f"请确认已安装 torch/albumentations 等依赖。"
            ) from e

        self.predictor = LesionPredictor(
            checkpoint_path=checkpoint_path,
            image_size=(image_size, image_size),
            conf_threshold=self.conf_threshold,
            device=self.device,
        )

    # ------------------------------------------------------------------ 主流程
    def run(self, image_path: str) -> dict:
        import cv2

        # ---- Stage1: 检测 + 疾病分类（现有模型）----
        det_result = self.predictor.predict(image_path)
        detections = det_result["detections"]
        detection_disease = det_result["disease"]  # 现有模型的直接疾病分类

        # ---- Stage2: 皮损计数 ----
        counts = count_lesions(detections)
        lesion_desc = explain_from_counts(counts)

        # ---- Stage3 信号A: 符号规则 ----
        rule_logits = score_from_counts(counts)
        rule_probs = softmax(rule_logits)

        # ---- Stage3 信号B: DermLIP 视觉零样本 ----
        dermlip_probs = None
        if self.dermlip is not None:
            img_bgr = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            from PIL import Image
            img_pil = Image.fromarray(
                np.ascontiguousarray(img_bgr[:, :, ::-1])
            )
            dermlip_probs = self.dermlip.zero_shot_disease(img_pil)
            # 若有病灶，逐 crop 零样本后与整图平均，增强病灶级信号
            if dermlip_probs is not None and detections:
                crop_probs = []
                for det in detections[:16]:  # 限制 crop 数量，控制 CPU 开销
                    crop = _crop_to_pil(img_bgr, det["bbox"])
                    if crop is None:
                        continue
                    p = self.dermlip.zero_shot_disease(crop)
                    if p:
                        crop_probs.append(p)
                if crop_probs:
                    keys = dermlip_probs.keys()
                    for k in keys:
                        vals = [p[k] for p in crop_probs] + [dermlip_probs[k]]
                        dermlip_probs[k] = float(np.mean(vals))

        # ---- 融合 ----
        if dermlip_probs is not None:
            final_probs = {
                d: self.alpha * rule_probs[d] + (1 - self.alpha) * dermlip_probs[d]
                for d in DISEASE_CLASSES
            }
        else:
            final_probs = rule_probs
            dermlip_probs = None

        pred = max(final_probs, key=final_probs.get)

        # ---- 文字诊断 ----
        diagnosis_text = self._build_text(
            pred, final_probs, counts, lesion_desc,
            rule_probs, dermlip_probs, detection_disease,
        )

        return {
            "image_path": str(image_path),
            "disease": pred,
            "disease_zh": DISEASE_CLASSES_ZH[pred],
            "confidence": float(final_probs[pred]),
            "probs": {d: float(final_probs[d]) for d in DISEASE_CLASSES},
            "signal_rule": {d: float(rule_probs[d]) for d in DISEASE_CLASSES},
            "signal_dermlip": ({d: float(dermlip_probs[d]) for d in DISEASE_CLASSES}
                               if dermlip_probs else None),
            "detection_model_disease": detection_disease.get("class"),
            "lesion_counts": counts,
            "lesion_desc": lesion_desc,
            "diagnosis_text": diagnosis_text,
            "n_detections": len(detections),
            "detections": detections,
        }

    # ------------------------------------------------------------------ 文本
    def _build_text(self, pred, final_probs, counts, lesion_desc,
                    rule_probs, dermlip_probs, detection_disease):
        lines = []
        n = sum(counts.values())
        lines.append(f"检测到 {n} 处皮损（{lesion_desc}）。")

        # 规则层解释
        rule_pred = max(rule_probs, key=rule_probs.get)
        if rule_pred == pred:
            lines.append(f"皮损谱规则判断指向{DISEASE_CLASSES_ZH[pred]}。")
        else:
            lines.append(
                f"皮损谱规则判断为{DISEASE_CLASSES_ZH[rule_pred]}，"
                f"与最终结果不一致。"
            )

        # DermLIP 层解释
        if dermlip_probs is not None:
            dl_pred = max(dermlip_probs, key=dermlip_probs.get)
            lines.append(
                f"DermLIP 视觉零样本倾向{DISEASE_CLASSES_ZH[dl_pred]}"
                f"（{dermlip_probs[dl_pred]:.0%}）。"
            )
        else:
            lines.append("DermLIP 不可用，本次仅依据皮损规则。")

        # 现有检测模型解释
        if detection_disease:
            det_cls = detection_disease.get("class", "")
            lines.append(
                f"现有检测模型分类为"
                f"{DISEASE_CLASSES_ZH.get(det_cls, det_cls)}"
                f"（{detection_disease.get('confidence', 0):.0%}）。"
            )

        lines.append(
            f"综合诊断：{DISEASE_CLASSES_ZH[pred]}"
            f"（置信度 {final_probs[pred]:.0%}）。"
        )
        return "".join(lines)
