"""
皮肤病分类的多任务模型。
组合了：
  1. backbone 特征提取（ConvNeXt/Swin/ResNet）
  2. 病灶检测分支（FPN + RPN + RoI）
  3. 疾病分类分支（全局池化 → FC）
  4. 融合：全局特征 + 病灶特征 → 最终疾病预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import build_backbone, backbone_supports_detection
from models.classification_head import DiseaseClassifier
from models.detection_head import build_detection_head
from config import MODEL_CONFIG, NUM_DISEASE_CLASSES, NUM_LESION_CLASSES, NUM_POSITION_CLASSES


class MultiTaskEczemaModel(nn.Module):
    """
    端到端的多任务皮肤病分析模型。

    两种模式：
      - detection_enabled=True: 同时执行病灶检测与疾病分类
      - detection_enabled=False: 仅做疾病分类，适合快速训练/调试
    """

    def __init__(
        self,
        backbone_name: str = MODEL_CONFIG["backbone"],
        pretrained: bool = MODEL_CONFIG["pretrained"],
        num_disease_classes: int = NUM_DISEASE_CLASSES,
        num_lesion_classes: int = NUM_LESION_CLASSES,
        detection_enabled: bool = MODEL_CONFIG["detection_enabled"],
        offline: bool = False,
        num_position_classes: int = NUM_POSITION_CLASSES,
        detection_head: str = MODEL_CONFIG["detection_head"],
    ):
        super().__init__()
        self.detection_enabled = detection_enabled

        # 1) 构建骨干网络，用于提取多尺度图像特征。
        #    对于 ConvNeXt / Swin / ResNet 等主干，返回的特征图通常包含多层 stage。
        self.backbone, self.feature_channels = build_backbone(
            name=backbone_name, pretrained=pretrained, offline=offline,
        )

        # 全局特征维度：对最高层特征做全局池化后，得到统一长度的向量。
        self.global_feat_dim = self.feature_channels[-1]

        # 2) 病灶检测分支：利用 FPN 提取从浅层到深层的多尺度特征，
        #    然后使用检测头生成候选框和类别分数。
        if detection_enabled:
            # 单分辨率 backbone（ViT/DINOv2）无法提供多尺度金字塔特征，
            # 不能用于检测头 —— 明确报错而非静默失败。
            if not backbone_supports_detection(backbone_name):
                raise ValueError(
                    f"Backbone '{backbone_name}' is single-resolution "
                    f"(classification-only) and cannot be used with a detection "
                    f"head. Pick a pyramidal backbone (convnext/swin/resnet/"
                    f"efficientnet) or disable detection."
                )
            # 取最后 4 个 stage 作为 FPN 的输入，兼顾细节与语义信息。
            det_in_channels = (
                self.feature_channels[-4:]
                if len(self.feature_channels) >= 4
                else self.feature_channels
            )
            self.detection_head = build_detection_head(
                name=detection_head,
                in_channels_list=det_in_channels,
                fpn_channels=MODEL_CONFIG["fpn_channels"],
                num_classes=num_lesion_classes,
            )
            self.lesion_feat_dim = MODEL_CONFIG["fpn_channels"]
        else:
            # 检测分支关闭时，直接使用骨干网络的全局池化结果模拟 lesion feature，
            # 以保持模型输出接口稳定并降低计算量。
            self.lesion_feat_dim = self.global_feat_dim

        # 3) 疾病分类分支：融合全局特征 + 病灶特征，用于最终病种判断。
        self.disease_classifier = DiseaseClassifier(
            in_features=self.global_feat_dim + self.lesion_feat_dim,
            num_classes=num_disease_classes,
        )

        # 4) 辅助分类头：仅使用全局特征进行疾病分类，
        #    作为额外监督信号，帮助稳定训练过程。
        self.global_classifier = DiseaseClassifier(
            in_features=self.global_feat_dim,
            num_classes=num_disease_classes,
        )

        # 5) 拍摄位置分类头：从全局特征预测 L/S/J/B 位置
        self.position_classifier = DiseaseClassifier(
            in_features=self.global_feat_dim,
            num_classes=num_position_classes,
        )

    def forward(self, images: torch.Tensor, targets: list[dict] = None) -> dict:
        """
        前向传播。

        参数：
            images: [B, C, H, W] 归一化图像张量。
            targets: 可选的训练目标字典列表。

        返回：
            字典，包含：
                - disease_logits: [B, num_disease_classes]
                - global_logits: [B, num_disease_classes]（辅助）
                - detection: 字典（boxes, labels, scores），若启用检测
                - losses: 损失字典（仅训练时）
        """
        # 先提取主干网络特征，再分别构造全局表征和病灶表征。
        backbone_features = self.backbone(images)

        # 对最深层特征进行全局平均池化，得到整张图的全局语义表示。
        stage_keys = sorted(backbone_features.keys())
        global_feat = backbone_features[stage_keys[-1]]

        global_pooled = F.adaptive_avg_pool2d(global_feat, (1, 1)).flatten(1)

        # 检测分支会输出各类病灶的框、标签和置信度，并生成 lesion feature。
        if self.detection_enabled:
            # 使用最后 4 个 stage 输入 FPN, 捕获不同分辨率的信息。
            stage_keys = sorted(backbone_features.keys())[-4:]
            detection_inputs = {k: backbone_features[k] for k in stage_keys}

            detection_output = self.detection_head(
                detection_inputs, images, targets
            )

            # 从 FPN 中提取病灶特征，供分类头融合使用。
            lesion_features = self.detection_head.get_lesion_features(
                detection_inputs, images
            )
        else:
            detection_output = {}
            # 关闭检测分支时，使用整图全局特征近似病灶特征，维持输出接口一致。
            lesion_features = F.adaptive_avg_pool2d(
                list(backbone_features.values())[-1], (1, 1)
            ).flatten(1)

        # 融合全局语义与病灶局部信息，形成更稳定的病种判别特征。
        fused_features = torch.cat([global_pooled, lesion_features], dim=1)

        # 进行最终疾病分类，并保留一个辅助全局分类头用于监督。
        disease_logits = self.disease_classifier(fused_features)
        global_logits = self.global_classifier(global_pooled)

        # 拍摄位置分类（基于全局特征）
        position_logits = self.position_classifier(global_pooled)

        result = {
            "disease_logits": disease_logits,
            "global_logits": global_logits,
            "position_logits": position_logits,
            "detection": detection_output,
        }

        if targets is not None:
            losses = {}
            # 检测损失
            if "losses" in detection_output:
                losses.update(detection_output["losses"])

            # 疾病分类损失（在 trainer 中计算）
            result["losses"] = losses

        return result

    def predict(self, images: torch.Tensor) -> dict:
        """
        推理模式：返回疾病预测与检测到的病灶。

        返回：
            字典，包含：
                - disease_probs: [B, num_classes] softmax 概率
                - disease_pred: [B] 预测类别索引
                - detection_boxes: [N, 4] 边界框列表
                - detection_labels: [N] 类别索引列表
                - detection_scores: [N] 置信度分数列表
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(images, targets=None)

        disease_probs = F.softmax(output["disease_logits"], dim=-1)
        disease_pred = disease_probs.argmax(dim=-1)

        position_probs = F.softmax(output["position_logits"], dim=-1)
        position_pred = position_probs.argmax(dim=-1)

        detection = output.get("detection", {})

        return {
            "disease_probs": disease_probs,
            "disease_pred": disease_pred,
            "position_probs": position_probs,
            "position_pred": position_pred,
            "detection_boxes": detection.get("boxes", []),
            "detection_labels": detection.get("labels", []),
            "detection_scores": detection.get("scores", []),
        }


def build_model(
    backbone_name: str = MODEL_CONFIG["backbone"],
    pretrained: bool = MODEL_CONFIG["pretrained"],
    detection_enabled: bool = MODEL_CONFIG["detection_enabled"],
    offline: bool = False,
    detection_head: str = MODEL_CONFIG["detection_head"],
) -> MultiTaskEczemaModel:
    """
    构建完整多任务模型的工厂函数。
    """
    return MultiTaskEczemaModel(
        backbone_name=backbone_name,
        pretrained=pretrained,
        detection_enabled=detection_enabled,
        offline=offline,
        detection_head=detection_head,
    )
