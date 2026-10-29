"""
基于 FPN + anchor-free 检测（FCOS 风格）的病灶检测头。

自包含实现 —— 不依赖 torchvision 的 RPN/RoI
（这些组件在不同 torchvision 版本之间存在 API 兼容性问题）。

设计：
  1. FPN 用于多尺度特征融合（torchvision.ops.FeaturePyramidNetwork）
  2. 每个 FPN 层级共享卷积头 → 类别 logits + 框回归 + centerness
  3. 训练：落在 GT 框内部的位置为正样本；根据框面积分配到 FPN 层级
     （小框 → 高分辨率层级，大框 → 低分辨率层级）。
  4. 推理：score = sqrt(cls_prob × centerness)，阈值过滤 + 批量 NMS。

参考文献：
  - FCOS: Fully Convolutional One-Stage Object Detection (Tian et al., 2019)
  - RetinaNet: Focal Loss for Dense Object Detection (Lin et al., 2017)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork, batched_nms

from config import NUM_LESION_CLASSES, MODEL_CONFIG


# ---------------------------------------------------------------------------
# 辅助模块：每个层级可学习的缩放因子（来自 FCOS）
# ---------------------------------------------------------------------------

class Scale(nn.Module):
    """每个 FPN 层级可学习的标量乘子，用于稳定训练。"""
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


# ---------------------------------------------------------------------------
# 应用于每个 FPN 层级的共享检测头
# ---------------------------------------------------------------------------

class SharedDetectionHead(nn.Module):
    """
    独立应用于每个 FPN 层级的共享检测头。

    每个空间位置预测：
      - class_logits: [num_lesion_classes]
      - bbox_deltas:  [4]  到图像空间中 (x1,y1,x2,y2) 的偏移量
      - centerness:   [1]  质量分数 ∈ [0,1]
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_convs: int = 4,
    ):
        super().__init__()
        # 共享卷积层
        layers = []
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=in_channels),
                nn.ReLU(inplace=True),
            ])
        self.convs = nn.Sequential(*layers)

        # 预测分支
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # 参数初始化（RetinaNet 风格的先验）
        for m in [self.cls_logits, self.bbox_pred, self.centerness]:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        # 先验概率偏置：π=0.01 → bias ≈ −4.595
        prior_prob = 0.01
        bias_init = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_init)


# ---------------------------------------------------------------------------
# 主检测头
# ---------------------------------------------------------------------------

class LesionDetectionHead(nn.Module):
    """
    带 FPN 的 anchor-free 病灶检测头。

    参数：
        in_channels_list: backbone 各 stage 的通道维度（最后 4 或 5 个）。
        fpn_channels: FPN 层的输出通道数。
        num_classes: 皮损类别数（无需背景类）。
        strides: 输入特征图相对于输入图像的步长。
        fg_iou_thresh: 正样本分配的 IoU 阈值（暂未使用；保留以便
                       未来兼容 ATSS 风格分配）。
    """

    def __init__(
        self,
        in_channels_list: list[int],
        fpn_channels: int = MODEL_CONFIG["fpn_channels"],
        num_classes: int = NUM_LESION_CLASSES,
        strides: list[int] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        num_fpn_levels = len(in_channels_list)

        # 特征金字塔网络（FPN）
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_channels,
            extra_blocks=None,
        )

        # 共享检测头（一个实例，应用到每个 FPN 层级）
        self.head = SharedDetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
        )

        # 每个层级可学习的缩放因子（有助于平衡各层级的损失）
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in range(num_fpn_levels)]
        )

        # 步长 —— 未提供时从 backbone 推断
        if strides is None:
            self.strides = [2 ** (i + 2) for i in range(num_fpn_levels)]
        else:
            self.strides = strides

        # 损失配置
        self.focal_alpha: float = 0.25
        self.focal_gamma: float = 2.0
        self.bbox_loss_beta: float = 1.0  # Smooth-L1 的 beta 参数

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(
        self,
        features: dict[str, torch.Tensor],
        images: torch.Tensor,
        targets: list[dict] | None = None,
    ) -> dict:
        """
        参数：
            features: backbone 特征字典（键形如 'stage1'..'stageN'）。
            images:  输入图像 [B, C, H, W]。
            targets: 可选的逐图像目标字典列表，包含
                     'boxes' [N_i, 4] 和 'labels' [N_i]。

        返回：
            字典，包含：
              - 'boxes':  [M_i, 4] 张量列表（推理）
              - 'labels': [M_i] 张量列表（推理）
              - 'scores': [M_i] 张量列表（推理）
              - 'losses': 损失标量字典（训练）
        """
        # --- FPN ---
        fpn_out = self.fpn(features)
        # 按分辨率排序（高 → 低）以保证层级顺序一致
        fpn_keys = sorted(fpn_out.keys())  # 例如 ['stage1','stage2','stage3','stage4']
        fpn_feats = [fpn_out[k] for k in fpn_keys]

        if targets is not None:
            return self._forward_train(fpn_feats, images, targets)

        return self._forward_inference(fpn_feats, images)

    # ------------------------------------------------------------------
    # 训练前向传播
    # ------------------------------------------------------------------

    def _forward_train(
        self,
        fpn_feats: list[torch.Tensor],
        images: torch.Tensor,
        targets: list[dict],
    ) -> dict:
        B = images.shape[0]
        device = images.device

        # 对每个 FPN 层级应用检测头
        all_cls_logits = []   # [B, C, H_i, W_i] 列表
        all_bbox_preds = []   # [B, 4, H_i, W_i] 列表
        all_centerness = []   # [B, 1, H_i, W_i] 列表

        for i, feat in enumerate(fpn_feats):
            cls_logits, bbox_pred, centerness = self._apply_head(feat, i)
            all_cls_logits.append(cls_logits)
            all_bbox_preds.append(bbox_pred)
            all_centerness.append(centerness)

        # 构建每个 FPN 层级的训练目标
        losses = self._compute_losses(
            all_cls_logits, all_bbox_preds, all_centerness,
            fpn_feats, images, targets,
        )
        return {"losses": losses}

    # ------------------------------------------------------------------
    # 推理前向传播
    # ------------------------------------------------------------------

    def _forward_inference(
        self,
        fpn_feats: list[torch.Tensor],
        images: torch.Tensor,
    ) -> dict:
        B = images.shape[0]
        device = images.device
        H_in, W_in = images.shape[-2:]

        all_boxes: list[list[torch.Tensor]] = [[] for _ in range(B)]
        all_scores: list[list[torch.Tensor]] = [[] for _ in range(B)]
        all_labels: list[list[torch.Tensor]] = [[] for _ in range(B)]

        for i, feat in enumerate(fpn_feats):
            stride = self.strides[i]
            cls_logits, bbox_pred, centerness = self._apply_head(feat, i)
            # cls_logits:  [B, C, H, W]
            # bbox_pred:   [B, 4, H, W]
            # centerness:  [B, 1, H, W]

            for b in range(B):
                boxes, scores, labels = self._decode_level(
                    cls_logits[b],
                    bbox_pred[b],
                    centerness[b],
                    stride,
                    (H_in, W_in),
                    score_thresh=0.05,
                )
                if boxes is not None:
                    all_boxes[b].append(boxes)
                    all_scores[b].append(scores)
                    all_labels[b].append(labels)

        # 合并各 FPN 层级并对每张图像应用 NMS
        result_boxes: list[torch.Tensor] = []
        result_scores: list[torch.Tensor] = []
        result_labels: list[torch.Tensor] = []

        for b in range(B):
            if not all_boxes[b]:
                result_boxes.append(torch.zeros((0, 4), device=device))
                result_scores.append(torch.zeros(0, device=device))
                result_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            boxes_b = torch.cat(all_boxes[b], dim=0)
            scores_b = torch.cat(all_scores[b], dim=0)
            labels_b = torch.cat(all_labels[b], dim=0)

            # NMS
            keep = batched_nms(boxes_b, scores_b, labels_b, iou_threshold=0.5)
            # 限制为 top-k
            keep = keep[:MODEL_CONFIG["box_detections_per_img"]]

            result_boxes.append(boxes_b[keep])
            result_scores.append(scores_b[keep])
            result_labels.append(labels_b[keep])

        return {
            "boxes": result_boxes,
            "labels": result_labels,
            "scores": result_scores,
        }

    # ------------------------------------------------------------------
    # 对单个 FPN 层级应用共享检测头
    # ------------------------------------------------------------------

    def _apply_head(
        self,
        feat: torch.Tensor,
        level_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对单个 FPN 层级应用共享卷积头 + 预测分支。

        返回：
            cls_logits:  [B, num_classes, H, W]
            bbox_pred:   [B, 4, H, W]   —— 到图像坐标中 (x1, y1, x2, y2) 的距离
            centerness:  [B, 1, H, W]
        """
        x = self.head.convs(feat)                      # [B, fpn_c, H, W]
        cls_logits = self.head.cls_logits(x)            # [B, C, H, W]
        bbox_pred = self.scales[level_idx](
            self.head.bbox_pred(x)
        )                                                # [B, 4, H, W]
        # bbox_pred 以像素为单位；用 exp 确保 l, t, r, b 为正值
        bbox_pred = torch.exp(bbox_pred.clamp(-9, 9))    # clamp 以保证数值稳定
        centerness = self.head.centerness(x)             # [B, 1, H, W]
        return cls_logits, bbox_pred, centerness

    # ------------------------------------------------------------------
    # 解码单个 FPN 层级的预测
    # ------------------------------------------------------------------

    def _decode_level(
        self,
        cls_logits: torch.Tensor,     # [C, H, W]
        bbox_pred: torch.Tensor,      # [4, H, W]
        centerness: torch.Tensor,     # [1, H, W]
        stride: int,
        image_size: tuple[int, int],
        score_thresh: float = 0.05,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        解码单个 FPN 层级对一张图像的预测。
        """
        C, H, W = cls_logits.shape
        H_in, W_in = image_size
        device = cls_logits.device

        # 计算每个位置的得分：sqrt(cls_prob * sigmoid(centerness))
        cls_probs = cls_logits.sigmoid()                         # [C, H, W]
        ctr = centerness.sigmoid().squeeze(0)                    # [H, W]
        scores = torch.sqrt(cls_probs * ctr.unsqueeze(0))        # [C, H, W]

        # 找出得分高于阈值的位置
        max_scores, max_classes = scores.max(dim=0)              # [H, W], [H, W]
        mask = max_scores > score_thresh
        if mask.sum() == 0:
            return None, None, None

        ys, xs = torch.where(mask)

        # 映射到图像坐标
        # 步长为 s 的特征图上的位置 (x, y) → 图像坐标
        # 使用 FCOS 约定：将 (cx, cy) 映射为每个网格单元的中心
        img_cx = (xs.float() + 0.5) * stride
        img_cy = (ys.float() + 0.5) * stride

        # 获取预测的距离：l, t, r, b(训练时按 stride 归一化,推理时乘回像素)
        l_pred = bbox_pred[0, ys, xs] * stride
        t_pred = bbox_pred[1, ys, xs] * stride
        r_pred = bbox_pred[2, ys, xs] * stride
        b_pred = bbox_pred[3, ys, xs] * stride

        x1 = img_cx - l_pred
        y1 = img_cy - t_pred
        x2 = img_cx + r_pred
        y2 = img_cy + b_pred

        # 裁剪到图像边界
        x1 = x1.clamp(0, W_in)
        y1 = y1.clamp(0, H_in)
        x2 = x2.clamp(0, W_in)
        y2 = y2.clamp(0, H_in)

        # 过滤无效框
        valid = (x2 > x1) & (y2 > y1)
        if valid.sum() == 0:
            return None, None, None

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)[valid]
        final_scores = max_scores[ys, xs][valid]
        final_labels = max_classes[ys, xs][valid]

        return boxes, final_scores, final_labels

    # ------------------------------------------------------------------
    # 损失计算
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        cls_logits_list: list[torch.Tensor],
        bbox_preds_list: list[torch.Tensor],
        centerness_list: list[torch.Tensor],
        fpn_feats: list[torch.Tensor],
        images: torch.Tensor,
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        """
        计算分类 + 回归 + centerness 损失。

        对每个 GT 框：
          1. 根据框面积分配到最佳的 FPN 层级。
          2. 中心落在框内的位置为正样本。
          3. 以 3:1 的比例采样负样本（不在任何框内的位置），
             使模型学会抑制误检。
        """
        B = images.shape[0]
        device = images.device
        num_levels = len(fpn_feats)
        num_neg_per_pos = 3  # 每个正样本对应的负样本数

        # 逐图像跨层级汇总所有预测
        flat_cls_targets: list[torch.Tensor] = []
        flat_cls_logits: list[torch.Tensor] = []
        flat_bbox_targets: list[torch.Tensor] = []
        flat_bbox_preds: list[torch.Tensor] = []
        flat_ctr_targets: list[torch.Tensor] = []
        flat_ctr_preds: list[torch.Tensor] = []

        for b in range(B):
            gt_boxes = targets[b].get("boxes", None)
            gt_labels = targets[b].get("labels", None)

            if gt_boxes is None or len(gt_boxes) == 0:
                continue

            # 为该图像构建各层级的正样本掩码
            level_pos_masks: list[torch.Tensor] = []
            level_pos_ys: list[torch.Tensor] = []
            level_pos_xs: list[torch.Tensor] = []
            level_targets: list[dict] = []

            for li in range(num_levels):
                H, W = fpn_feats[li].shape[-2:]
                pos_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
                pos_data: dict[int, dict] = {}  # (y,x) → 目标数据
                level_pos_masks.append(pos_mask)
                level_pos_ys.append([])
                level_pos_xs.append([])
                level_targets.append(pos_data)

            # 收集正样本
            for gi, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                label_idx = label.item() if isinstance(label, torch.Tensor) else label

                levels = _assign_levels(
                    (x2 - x1) * (y2 - y1),
                    self.strides,
                )

                for li in levels:
                    H, W = fpn_feats[li].shape[-2:]
                    stride = self.strides[li]

                    gx = (torch.arange(W, device=device).float() + 0.5) * stride
                    gy = (torch.arange(H, device=device).float() + 0.5) * stride

                    in_x = (gx >= x1) & (gx <= x2)
                    in_y = (gy >= y1) & (gy <= y2)
                    mask = in_y.unsqueeze(1) & in_x.unsqueeze(0)

                    if mask.sum() == 0:
                        continue

                    py, px = torch.where(mask)
                    cx = gx[px]
                    cy = gy[py]

                    # 回归目标按 stride 归一化(格子单位):像素距离 / stride。
                    # 这样目标量级 O(1)、与输入分辨率无关,避免像素单位下
                    # 回归损失随分辨率暴涨、淹没分类/centerness 头。
                    # centerness 是 min/max 比值,不受除以 stride 影响。
                    l_tgt = ((cx - x1) / stride).clamp(min=0)
                    t_tgt = ((cy - y1) / stride).clamp(min=0)
                    r_tgt = ((x2 - cx) / stride).clamp(min=0)
                    b_tgt = ((y2 - cy) / stride).clamp(min=0)

                    ctr_tgt = torch.sqrt(
                        (torch.min(l_tgt, r_tgt) / torch.max(l_tgt, r_tgt).clamp(min=1e-6))
                        * (torch.min(t_tgt, b_tgt) / torch.max(t_tgt, b_tgt).clamp(min=1e-6))
                    )

                    cls_tgt = torch.zeros(self.num_classes, device=device)
                    cls_tgt[label_idx] = 1.0

                    # 保存正样本数据并更新掩码
                    bbox_tgt = torch.stack([l_tgt, t_tgt, r_tgt, b_tgt], dim=1)
                    for j in range(len(py)):
                        key = (py[j].item(), px[j].item())
                        # 若已被另一个框标记为正样本，保留第一次分配
                        if key not in level_targets[li]:
                            level_targets[li][key] = {
                                "cls": cls_tgt,
                                "bbox": bbox_tgt[j],
                                "ctr": ctr_tgt[j],
                            }

                    # 更新各层级的正样本掩码（所有框的并集）
                    # 需要用唯一索引重建掩码
                    # 处理完所有框后再收集唯一位置

            # --- 处理完该图像的所有框后，收集正样本 ---
            for li in range(num_levels):
                pos_data = level_targets[li]
                if not pos_data:
                    continue

                H, W = fpn_feats[li].shape[-2:]
                # 构建所有正样本位置的掩码，用于负样本采样
                pos_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
                for (py, px) in pos_data:
                    pos_mask[py, px] = True

                # 将正样本加入扁平列表
                for (py, px), data in pos_data.items():
                    flat_cls_targets.append(data["cls"])
                    flat_cls_logits.append(cls_logits_list[li][b, :, py, px])
                    flat_bbox_targets.append(data["bbox"])
                    flat_bbox_preds.append(bbox_preds_list[li][b, :, py, px])
                    flat_ctr_targets.append(data["ctr"])
                    flat_ctr_preds.append(centerness_list[li][b, 0, py, px])

                # --- 从该层级采样负样本 ---
                n_pos = len(pos_data)
                n_neg = n_pos * num_neg_per_pos

                # 候选负样本：不在任何正样本掩码中的位置
                neg_mask = ~pos_mask
                neg_ys, neg_xs = torch.where(neg_mask)
                n_available = len(neg_ys)

                if n_available > 0:
                    n_sample = min(n_neg, n_available)
                    neg_indices = torch.randperm(n_available, device=device)[:n_sample]
                    neg_py = neg_ys[neg_indices]
                    neg_px = neg_xs[neg_indices]

                    neg_cls_target = torch.zeros(self.num_classes, device=device)
                    for j in range(len(neg_py)):
                        flat_cls_targets.append(neg_cls_target)
                        flat_cls_logits.append(
                            cls_logits_list[li][b, :, neg_py[j].item(), neg_px[j].item()]
                        )
                        # 负样本的占位 bbox/ctr 目标（不用于 reg/ctr 损失）
                        flat_bbox_targets.append(torch.zeros(4, device=device))
                        flat_bbox_preds.append(
                            bbox_preds_list[li][b, :, neg_py[j].item(), neg_px[j].item()]
                        )
                        flat_ctr_targets.append(torch.tensor(0.0, device=device))
                        flat_ctr_preds.append(
                            centerness_list[li][b, 0, neg_py[j].item(), neg_px[j].item()]
                        )

        # 若整个 batch 无正样本，返回零损失
        if len(flat_cls_targets) == 0:
            return {
                "loss_det_cls": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_det_reg": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_det_ctr": torch.tensor(0.0, device=device, requires_grad=True),
            }

        # 堆叠
        cls_targets_t = torch.stack(flat_cls_targets)      # [N, C]
        cls_logits_t = torch.stack(flat_cls_logits)         # [N, C]
        bbox_targets_t = torch.stack(flat_bbox_targets)     # [N, 4]
        bbox_preds_t = torch.stack(flat_bbox_preds)         # [N, 4]
        ctr_targets_t = torch.stack(flat_ctr_targets)       # [N]
        ctr_preds_t = torch.stack(flat_ctr_preds)           # [N]

        # 将正负样本分开，用于回归与 centerness 损失
        is_pos = cls_targets_t.sum(dim=1) > 0  # [N] —— 正样本在 one-hot 中有一个 1
        n_pos = is_pos.sum().item()

        # 分类用 Focal Loss（对所有样本：正 + 负）
        loss_cls = _focal_loss(
            cls_logits_t,
            cls_targets_t,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

        # 回归 + centerness 损失仅作用于正样本
        if n_pos > 0:
            loss_reg = F.smooth_l1_loss(
                bbox_preds_t[is_pos],
                bbox_targets_t[is_pos],
                beta=self.bbox_loss_beta,
            )
            # 用 with_logits 版本：AMP autocast 下安全，且内部 log-sum-exp 数值更稳定
            loss_ctr = F.binary_cross_entropy_with_logits(
                ctr_preds_t[is_pos].float(),
                ctr_targets_t[is_pos].float(),
            )
        else:
            loss_reg = torch.tensor(0.0, device=device, requires_grad=True)
            loss_ctr = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            "loss_det_cls": loss_cls,
            "loss_det_reg": loss_reg,
            "loss_det_ctr": loss_ctr,
        }

    # ------------------------------------------------------------------
    # 用于疾病分类融合的病灶特征
    # ------------------------------------------------------------------

    def get_lesion_features(
        self,
        features: dict[str, torch.Tensor],
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取病灶级特征用于全局分类融合。

        对最粗的 FPN 层级（感受野最大 —— 语义特征最好）
        做池化，得到每张图像的固定长度向量。
        """
        fpn_out = self.fpn(features)
        fpn_keys = sorted(fpn_out.keys())
        # 使用最粗（最低分辨率）层级 —— 最深层的语义
        coarsest = fpn_out[fpn_keys[-1]]  # [B, C, H, W]
        pooled = F.adaptive_avg_pool2d(coarsest, (1, 1)).flatten(1)
        return pooled


# =========================================================================
# 辅助函数
# =========================================================================

def _assign_levels(
    box_area: torch.Tensor | float,
    strides: list[int],
) -> list[int]:
    """
    根据框面积将框分配到最佳 FPN 层级（FCOS 风格）。

    目标层级为：floor(log₂(√area) - 3)，并裁剪到有效范围。
    这确保每个框被分配到唯一一个层级，减少训练歧义。

    对于我们的 strides [4, 8, 16, 32]：
      √area <  32  → stride 4
      √area <  64  → stride 8
      √area < 128  → stride 16
      √area ≥ 128  → stride 32
    """
    if isinstance(box_area, torch.Tensor):
        box_area = box_area.item()

    sqrt_area = math.sqrt(max(box_area, 1.0))

    # FCOS 层级分配:level = floor(log2(√area) - offset)。
    # 病灶很小(512 下 √area 约 5~22px),若按标准 offset=2,中等框会被分到
    # stride 16/32,在那里只占 ~1 个格点、正样本极少甚至为 0。
    # 这里用 offset=4 把框推向更精细层级,使每个框在其分配层占 ~4+ 个格点:
    #   √area <  16 → stride 4
    #   16 ≤ √area < 32 → stride 8
    #   32 ≤ √area < 64 → stride 16
    #   ≥ 64  → stride 32
    level = int(math.floor(math.log2(sqrt_area + 1e-6) - 4))
    level = max(0, min(level, len(strides) - 1))

    return [level]


def build_detection_head(
    name: str,
    in_channels_list: list[int],
    fpn_channels: int = MODEL_CONFIG["fpn_channels"],
    num_classes: int = NUM_LESION_CLASSES,
    strides: list[int] | None = None,
) -> nn.Module:
    """
    检测头工厂。所有检测头实现统一接口：
      - forward(features, images, targets) → {"losses"} 或 {"boxes","labels","scores"}
      - get_lesion_features(features, images) → [B, C]

    可选值：'fcos'、'faster_rcnn'、'cascade_rcnn'、'yolov8'、'rtdetr'、
            'transunet'、'transcascade'、'attention_unet'、'mamba'、
            'sam_adapter'。
    """
    name = name.lower()
    if name in ("fcos",):
        return LesionDetectionHead(
            in_channels_list, fpn_channels, num_classes, strides,
        )
    if name in ("faster_rcnn", "fasterrcnn", "rcnn", "faster-rcnn"):
        from models.detection_faster_rcnn import FasterRCNNHead
        return FasterRCNNHead(in_channels_list, fpn_channels, num_classes)
    if name in ("cascade_rcnn", "cascade", "cascade-rcnn"):
        from models.detection_faster_rcnn import CascadeRCNNHead
        return CascadeRCNNHead(in_channels_list, fpn_channels, num_classes)
    if name in ("yolov8", "yolo", "yolo_v8"):
        from models.detection_yolov8 import YOLOv8Head
        return YOLOv8Head(in_channels_list, fpn_channels, num_classes)
    if name in ("rtdetr", "detr", "rt_detr"):
        from models.detection_rtdetr import RTDETRHead
        return RTDETRHead(in_channels_list, fpn_channels, num_classes)
    if name in ("transunet", "trans_unet", "trans-unet"):
        from models.transunet_head import TransUNetHead
        return TransUNetHead(in_channels_list, fpn_channels, num_classes)
    if name in ("transcascade", "trans_cascade", "trans-cascade"):
        from models.transunet_head import TransCASCADEHead
        return TransCASCADEHead(in_channels_list, fpn_channels, num_classes)
    if name in ("attention_unet", "attentionunet", "attn_unet", "attention-unet"):
        from models.attention_unet_head import AttentionUNetHead
        return AttentionUNetHead(in_channels_list, fpn_channels, num_classes)
    if name in ("mamba",):
        from models.mamba_head import MambaHead
        return MambaHead(in_channels_list, fpn_channels, num_classes)
    if name in ("sam_adapter", "samadapter", "sam-adapter"):
        from models.sam_adapter_head import SAMAdapterHead
        return SAMAdapterHead(in_channels_list, fpn_channels, num_classes)
    raise ValueError(f"Unknown detection head: {name}")


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float | torch.Tensor = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    多标签分类的 Sigmoid Focal Loss。

    参数：
        logits:  [N, C] 原始 logits。
        targets: [N, C] one-hot（或多 hot）目标。
        alpha:   正样本的类别平衡权重。可为标量，或形状 [C] 的按类别张量
                 （按类别分布自动计算，缓解长尾）。
        gamma:   聚焦参数。
    """
    prob = logits.sigmoid()
    prob = prob.clamp(1e-6, 1.0 - 1e-6)

    # 二值交叉熵（BCE）
    ce_pos = -targets * torch.log(prob)
    ce_neg = -(1.0 - targets) * torch.log(1.0 - prob)

    # Focal 权重
    p_t = targets * prob + (1.0 - targets) * (1.0 - prob)
    focal_weight = (1.0 - p_t) ** gamma

    # Alpha 平衡（支持按类别的张量 alpha）
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to(logits.device)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)

    loss = alpha_t * focal_weight * (ce_pos + ce_neg)
    return loss.sum() / max(targets.sum(), 1.0)
