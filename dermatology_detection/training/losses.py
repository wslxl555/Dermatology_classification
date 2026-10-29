"""
多任务皮肤疾病分类的损失函数。
包含用于处理类别不平衡的 Focal Loss 以及 label smoothing。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    用于处理类别不平衡的 Focal Loss。

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    其中 p_t 是模型对真实类别的估计概率。
    """

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        参数:
            alpha: 形状为 [num_classes] 的类别权重张量。
            gamma: 聚焦参数。越大越聚焦于难分样本。
            reduction: 'mean'、'sum' 或 'none'。
            label_smoothing: 标签平滑系数(0 表示不平滑)。
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        参数:
            logits: [B, C] 原始 logits。
            targets: [B] 类别索引。

        返回:
            标量损失。
        """
        num_classes = logits.size(-1)

        # 应用 label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(log_probs, targets, reduction="none")

        # Focal 缩放
        with torch.no_grad():
            probs = torch.exp(-loss)  # p_t
            focal_weight = (1 - probs) ** self.gamma

        loss = focal_weight * loss

        # 类别权重
        if self.alpha is not None:
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)
            alpha_weight = self.alpha[targets]
            loss = alpha_weight * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """
    多任务学习的组合损失:
      L_total = λ_disease * L_disease + λ_global * L_global
              + λ_detection * L_detection
    """

    def __init__(
        self,
        num_disease_classes: int = 3,
        class_weights: torch.Tensor = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        loss_disease_weight: float = 1.0,
        loss_global_weight: float = 0.3,
        loss_detection_weight: float = 0.5,
    ):
        super().__init__()

        self.disease_loss = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            reduction="mean",
            label_smoothing=label_smoothing,
        )
        self.global_loss = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            reduction="mean",
            label_smoothing=label_smoothing,
        )

        self.loss_disease_weight = loss_disease_weight
        self.loss_global_weight = loss_global_weight
        self.loss_detection_weight = loss_detection_weight

    def forward(
        self,
        disease_logits: torch.Tensor,
        global_logits: torch.Tensor,
        disease_targets: torch.Tensor,
        detection_losses: dict = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        计算多任务总损失。

        参数:
            disease_logits: [B, num_disease] 融合后的分类 logits。
            global_logits: [B, num_disease] 仅全局分支的分类 logits。
            disease_targets: [B] 疾病类别索引。
            detection_losses: 来自 RPN/RoI 头的检测损失字典。

        返回:
            total_loss: 标量张量。
            loss_dict: 用于日志记录的各个损失分量。
        """
        loss_disease = self.disease_loss(disease_logits, disease_targets)
        loss_global = self.global_loss(global_logits, disease_targets)

        loss_dict = {
            "loss_disease": loss_disease.item(),
            "loss_global": loss_global.item(),
        }

        total_loss = (
            self.loss_disease_weight * loss_disease +
            self.loss_global_weight * loss_global
        )

        if detection_losses:
            loss_detection = sum(v for v in detection_losses.values())
            total_loss = total_loss + self.loss_detection_weight * loss_detection
            loss_dict["loss_detection"] = loss_detection.item()

        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict


def compute_class_weights(samples_per_class: list[int]) -> torch.Tensor:
    """
    计算用于 Focal Loss alpha 的逆频率类别权重。

    参数:
        samples_per_class: 每个疾病类别的样本数量列表。

    返回:
        权重张量,归一化到总和为 num_classes。
    """
    n = len(samples_per_class)
    total = sum(samples_per_class)
    weights = [total / (n * max(c, 1)) for c in samples_per_class]
    return torch.tensor(weights, dtype=torch.float32)
