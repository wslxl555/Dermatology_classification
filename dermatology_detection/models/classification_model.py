"""
纯疾病分类模型。

只做疾病三分类（湿疹 / 大疱型类天疱疮 / 荨麻疹），不涉及任何检测分支。
结构：backbone → 全局平均池化 → MLP 分类头 → 3 类 logits。

与多任务模型的关键区别：推理与验证阶段**只需要图像本身**，完全不读取
.json 标注文件（疾病标签来自图片所在目录，而非标注文件）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import build_backbone
from models.classification_head import DiseaseClassifier
from config import NUM_DISEASE_CLASSES, MODEL_CONFIG


class DiseaseClassificationModel(nn.Module):
    """
    端到端的纯疾病分类模型。

    参数:
        backbone_name: 主干网络名称（convnext_tiny / resnet50 / swin_t 等）。
        pretrained: 是否使用 ImageNet 预训练权重。
        num_classes: 疾病类别数（默认 3）。
        offline: 若为 True，跳过网络访问，仅使用缓存权重。
        hidden_dim / dropout: 分类头 MLP 的超参数（默认取 config 中的值）。
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = NUM_DISEASE_CLASSES,
        offline: bool = False,
        hidden_dim: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # 1) 特征提取 backbone
        self.backbone, self.feature_channels = build_backbone(
            name=backbone_name, pretrained=pretrained, offline=offline,
        )

        # 2) 全局特征维度 = 最深 stage 的通道数（convnext_tiny 为 768）
        self.feat_dim = self.feature_channels[-1]

        # 3) 分类头（LayerNorm → Linear → GELU → Dropout → ... → logits）
        head_kwargs = {}
        if hidden_dim is not None:
            head_kwargs["hidden_dim"] = hidden_dim
        if dropout is not None:
            head_kwargs["dropout"] = dropout
        self.head = DiseaseClassifier(
            self.feat_dim, num_classes=num_classes, **head_kwargs,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        参数:
            images: [B, C, H, W] 归一化图像张量。

        返回:
            logits: [B, num_classes] 疾病分类原始 logits。
        """
        # 提取 backbone 特征，取最深 stage 做全局池化。
        features = self.backbone(images)
        stage_keys = sorted(features.keys())
        global_feat = features[stage_keys[-1]]
        pooled = F.adaptive_avg_pool2d(global_feat, (1, 1)).flatten(1)
        return self.head(pooled)


def build_classification_model(
    backbone_name: str = "convnext_tiny",
    pretrained: bool = True,
    num_classes: int = NUM_DISEASE_CLASSES,
    offline: bool = False,
) -> DiseaseClassificationModel:
    """构建纯疾病分类模型的工厂函数。"""
    return DiseaseClassificationModel(
        backbone_name=backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        offline=offline,
    )
