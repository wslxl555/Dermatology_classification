"""
皮肤疾病分类的评估指标。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class MetricsTracker:
    """跨 batch 跟踪并计算分类指标。"""

    def __init__(self, num_classes: int = 3, class_names: list[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []

    def update(self, logits: np.ndarray, targets: np.ndarray):
        """
        参数:
            logits: [B, C] 原始 logits。
            targets: [B] 类别索引。
        """
        from scipy.special import softmax
        probs = softmax(logits, axis=1)
        preds = probs.argmax(axis=1)

        self.all_preds.extend(preds.tolist())
        self.all_targets.extend(targets.tolist())
        self.all_probs.extend(probs.tolist())

    def compute(self) -> dict:
        """
        计算所有指标。
        返回包含 accuracy、f1(macro + 每类)、AUC 等的字典。
        """
        if not self.all_preds:
            return {}

        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        probs = np.array(self.all_probs)

        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(targets, preds, average="weighted", zero_division=0),
            "macro_precision": precision_score(targets, preds, average="macro", zero_division=0),
            "macro_recall": recall_score(targets, preds, average="macro", zero_division=0),
        }

        # 每个类别的 F1
        per_class = f1_score(targets, preds, average=None, zero_division=0)
        for i, f1 in enumerate(per_class):
            name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            metrics[f"f1_{name}"] = f1

        # AUC-ROC(one-vs-rest)
        try:
            if probs.shape[1] >= 2:
                # 计算 AUC 需要每个类别至少有一个正样本
                present_classes = set(targets)
                if len(present_classes) >= 2:
                    metrics["auc_roc_ovr"] = roc_auc_score(
                        targets, probs, multi_class="ovr", average="macro"
                    )
        except ValueError:
            metrics["auc_roc_ovr"] = float("nan")

        # 混淆矩阵
        metrics["confusion_matrix"] = confusion_matrix(targets, preds).tolist()

        return metrics

    def get_classification_report(self) -> str:
        """以字符串形式获取 sklearn 分类报告。"""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        return classification_report(
            targets, preds, target_names=self.class_names, zero_division=0
        )
