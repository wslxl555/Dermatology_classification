"""
疾病分类头。
从全局图像特征中产出 4 类的疾病预测结果。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_DISEASE_CLASSES, MODEL_CONFIG


class DiseaseClassifier(nn.Module):
    """
    全局疾病分类器。

    接收池化后的 backbone 特征,预测 4 种疾病类别之一:
    eczema、contact_dermatitis、bullous_pemphigoid、urticaria。
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = MODEL_CONFIG["global_hidden_dim"],
        num_classes: int = NUM_DISEASE_CLASSES,
        dropout: float = MODEL_CONFIG["dropout"],
    ):
        super().__init__()

        # 多层感知机分类器:LayerNorm → Linear → GELU → Dropout → ... → logits
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        参数:
            features: [B, in_features] 全局池化后的特征。

        返回:
            logits: [B, num_classes] 原始 logits。
        """
        return self.classifier(features)
