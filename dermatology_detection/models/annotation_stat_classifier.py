"""
基于皮损标注统计信息的疾病分类器（两阶段流程的第二阶段）。

第一阶段（检测模型）对图片自动标注，输出 box + annotation（皮损类别）；
本模型学习"各类 annotation 的统计信息"与"各类疾病之间的关系"：
以一张图片的皮损类型统计直方图（15 维，归一化计数）为输入，输出 3 类疾病。

结构：Linear(15 → hidden) → GELU → Dropout → Linear(hidden → 3)。
两层线性权重可复合为 15×3 的"皮损类型 → 疾病"关联矩阵，用于解释
哪些皮损类型对哪种疾病最具判别性（即 annotation 与疾病的关系）。
"""

import torch
import torch.nn as nn

from config import NUM_LESION_CLASSES, NUM_DISEASE_CLASSES


class AnnotationStatClassifier(nn.Module):
    """
    皮损标注统计直方图 → 疾病分类的轻量 MLP。

    参数:
        num_lesion_classes: 皮损类别数（15）。
        num_diseases: 疾病类别数（3）。
        hidden_dim: 隐层维度。
        dropout: 隐层 dropout 概率。
    """

    def __init__(
        self,
        num_lesion_classes: int = NUM_LESION_CLASSES,
        num_diseases: int = NUM_DISEASE_CLASSES,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_lesion_classes = num_lesion_classes
        self.num_diseases = num_diseases
        self.hidden_dim = hidden_dim

        # 第一层：皮损统计直方图 → 隐层
        self.proj = nn.Linear(num_lesion_classes, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # 第二层：隐层 → 疾病 logits
        self.head = nn.Linear(hidden_dim, num_diseases)

        for m in (self.proj, self.head):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """
        参数:
            hist: [B, num_lesion_classes] 皮损统计直方图（归一化计数/占比）。

        返回:
            logits: [B, num_diseases] 疾病分类 logits。
        """
        x = self.act(self.proj(hist))
        x = self.drop(x)
        return self.head(x)

    def association_matrix(self) -> torch.Tensor:
        """
        返回 [num_lesion_classes, num_diseases] 的"皮损类型 → 疾病"关联矩阵。

        由两层线性权重复合得到：
            proj.weight: [hidden, 15]  →  转置后 [15, hidden]
            head.weight:  [3, hidden]  →  转置后 [hidden, 3]
            复合: [15, hidden] @ [hidden, 3] = [15, 3]

        数值（经 softmax 归一化后）表示"该皮损类型对某疾病判别贡献的强弱"。
        """
        return self.proj.weight.t() @ self.head.weight.t()


def build_annotation_classifier(
    num_lesion_classes: int = NUM_LESION_CLASSES,
    num_diseases: int = NUM_DISEASE_CLASSES,
    hidden_dim: int = 64,
    dropout: float = 0.3,
) -> AnnotationStatClassifier:
    """构建皮损统计 → 疾病分类器的工厂函数。"""
    return AnnotationStatClassifier(
        num_lesion_classes=num_lesion_classes,
        num_diseases=num_diseases,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
