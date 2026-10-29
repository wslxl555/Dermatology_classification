"""
两阶段检测头（自包含纯 PyTorch 实现）：Faster R-CNN 与 Cascade R-CNN。

设计：
  1. FPN 多尺度特征融合（torchvision.ops.FeaturePyramidNetwork）。
  2. RPN 在每个 FPN 层级生成 anchor，预测 objectness + 框回归，输出 proposals。
  3. RoIAlign（MultiScaleRoIAlign）将 proposals 池化为固定尺寸特征。
  4. RoI 头：分类（num_classes + 背景）+ 类无关框回归。
  5. Cascade R-CNN：3 个级联 RoI 阶段，IoU 阈值 [0.5, 0.6, 0.7]，
     每阶段用上一阶段的精修框作为输入。

仅复用稳定的 torchvision.ops 原语（roi_align / nms / box_iou / clip 等），
避免依赖不稳定的 RPN/RoI 高层 API。所有头共享与 FCOS 相同的接口。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import (
    FeaturePyramidNetwork, MultiScaleRoIAlign, nms, batched_nms,
    box_iou, clip_boxes_to_image, remove_small_boxes, box_area,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator


# ---------------------------------------------------------------------------
# 框编码/解码（Faster R-CNN 约定，类无关）
# ---------------------------------------------------------------------------

def _encode_boxes(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """gt/anchors 均为 [N, 4] (x1,y1,x2,y2) → deltas [N, 4]。"""
    ex_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    ex_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
    ex_w = anchors[:, 2] - anchors[:, 0] + 1.0
    ex_h = anchors[:, 3] - anchors[:, 1] + 1.0

    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0

    dx = (gt_cx - ex_cx) / ex_w
    dy = (gt_cy - ex_cy) / ex_h
    dw = torch.log(gt_w / ex_w)
    dh = torch.log(gt_h / ex_h)
    return torch.stack([dx, dy, dw, dh], dim=1)


def _decode_boxes(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """deltas/anchors 均为 [N, 4] → 解码框 [N, 4]。"""
    ex_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    ex_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
    ex_w = anchors[:, 2] - anchors[:, 0] + 1.0
    ex_h = anchors[:, 3] - anchors[:, 1] + 1.0

    cx = deltas[:, 0] * ex_w + ex_cx
    cy = deltas[:, 1] * ex_h + ex_cy
    w = torch.exp(deltas[:, 2]) * ex_w
    h = torch.exp(deltas[:, 3]) * ex_h

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=1)


# ---------------------------------------------------------------------------
# RPN 头
# ---------------------------------------------------------------------------

class RPNHead(nn.Module):
    """共享 3x3 conv + objectness + 框回归的 RPN 头。"""

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        for m in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, features: list[torch.Tensor]):
        """返回 (logits_list, bbox_list)，每个元素 [B, A, H, W] / [B, 4A, H, W]。"""
        logits_list, bbox_list = [], []
        for feat in features:
            t = F.relu(self.conv(feat))
            logits_list.append(self.cls_logits(t))
            bbox_list.append(self.bbox_pred(t))
        return logits_list, bbox_list


# ---------------------------------------------------------------------------
# RoI 头（全连接）
# ---------------------------------------------------------------------------

class RoIHead(nn.Module):
    """对 RoI 池化特征做分类（num_classes+1）与类无关框回归。"""

    def __init__(self, in_channels: int, num_classes: int, representation_size: int = 1024):
        super().__init__()
        self.fc6 = nn.Linear(in_channels * 7 * 7, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(representation_size, num_classes + 1)  # + 背景
        self.bbox_pred = nn.Linear(representation_size, 4)               # 类无关
        for m in [self.fc6, self.fc7, self.cls_score, self.bbox_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """x: [K, C, 7, 7] → (cls_logits [K, C+1], bbox_deltas [K, 4])。"""
        x = x.flatten(1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return self.cls_score(x), self.bbox_pred(x)


# ---------------------------------------------------------------------------
# 两阶段基类（Faster R-CNN 共享逻辑）
# ---------------------------------------------------------------------------

class _TwoStageHead(nn.Module):
    def __init__(self, in_channels_list, fpn_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels

        # FPN（输入 keys 归一化为 0..N-1）
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=fpn_channels,
        )
        self.num_levels = len(in_channels_list)

        # anchors：每层 3 个尺寸 × 3 个长宽比，尺寸按 1.5× 递增、相邻层首尾重叠，
        # 消除“尺寸空洞”。病灶在 512px 下以小结节/斑疹为主（中位约 18×22px，
        # 49% 最小边 <16px）。原先每层单尺寸 8/16/32/64 有 2× 跳变，叠加 RPN
        # 正样本 IoU 阈值 0.7，中位病灶(~20px)落在 16~24px 的空洞里，拿不到
        # 正样本 → roi_reg 恒 0、mAP≈0。加密为每层 3 尺寸，连续覆盖 8~144px。
        sizes = tuple(
            tuple(round(8 * (2 ** lvl) * (1.5 ** j)) for j in range(3))
            for lvl in range(self.num_levels)
        )
        self.anchor_generator = AnchorGenerator(
            sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * self.num_levels,
        )
        self.num_anchors = self.anchor_generator.num_anchors_per_location()[0]

        # RPN
        self.rpn_head = RPNHead(fpn_channels, self.num_anchors)

        # RoI 池化
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=[str(i) for i in range(self.num_levels)],
            output_size=7, sampling_ratio=2,
        )

    # -- 特征组织 ---------------------------------------------------------
    def _fpn_features(self, features: dict) -> list[torch.Tensor]:
        fpn_out = self.fpn(features)
        keys = sorted(fpn_out.keys())
        return [fpn_out[k] for k in keys]

    def _fpn_ordered(self, feats: list[torch.Tensor]) -> dict:
        return {"%d" % i: f for i, f in enumerate(feats)}

    def _grid_sizes_and_strides(self, feats, images):
        """返回 (grid_sizes, strides)。"""
        grid_sizes = [(f.shape[-2], f.shape[-1]) for f in feats]
        img_h, img_w = images.shape[-2:]
        strides = [img_h // gs[0] for gs in grid_sizes]
        return grid_sizes, strides

    def _anchors(self, feats, images):
        grid_sizes, strides = self._grid_sizes_and_strides(feats, images)
        device = images.device
        # 修复 device 不匹配：torchvision 的 AnchorGenerator.cell_anchors 在该版本
        # 不随 .to() 移动（落在 cpu），导致 grid_anchors 内 arange(cpu) 与
        # stride(cuda) 相乘报错。改用纯 float 的 stride（无 device），并把最终
        # 生成的 anchor 显式移到输入设备。
        strides_t = [[float(s), float(s)] for s in strides]
        per_level = self.anchor_generator.grid_anchors(grid_sizes, strides_t)
        anchors = torch.cat(per_level, dim=0).to(device)  # [N_total, 4] 展平
        return anchors, strides

    # -- RPN 前向 ---------------------------------------------------------
    def _rpn_forward(self, feats):
        """返回 (objectness [B, N_anchors], bbox_deltas [B, N_anchors, 4], anchors [N_anchors, 4])。"""
        logits_list, bbox_list = self.rpn_head(feats)

        objectness = []
        bbox_deltas = []
        for i in range(len(logits_list)):
            # [B, A, H, W] → [B, H*W*A]
            obj = logits_list[i].permute(0, 2, 3, 1).flatten(1)
            # [B, 4A, H, W] → [B, H*W*A, 4]
            bb = bbox_list[i].permute(0, 2, 3, 1).reshape(
                bbox_list[i].shape[0], -1, 4)
            objectness.append(obj)
            bbox_deltas.append(bb)
        objectness = torch.cat(objectness, dim=1)
        bbox_deltas = torch.cat(bbox_deltas, dim=1)
        return objectness, bbox_deltas

    # -- 匹配 anchor → GT -------------------------------------------------
    def _match_anchors(self, anchors, gt_boxes, pos_thresh=0.5, neg_thresh=0.3):
        """返回 (labels [N], matched_gt [N,4])，label ∈ {1,0,-1}。"""
        device = anchors.device
        n = anchors.shape[0]
        labels = torch.full((n,), -1, dtype=torch.long, device=device)
        matched_gt = torch.zeros((n, 4), device=device)

        if gt_boxes.numel() == 0:
            labels[torch.randperm(n, device=device)[:1]] = 1  # 占位正样本
            return labels, matched_gt

        iou = box_iou(anchors, gt_boxes)  # [N, M]
        max_iou, argmax_iou = iou.max(dim=1)

        labels[max_iou < neg_thresh] = 0
        labels[max_iou >= pos_thresh] = 1
        # 每个 GT 的最佳 anchor 强制为正
        for j in range(gt_boxes.shape[0]):
            best = iou[:, j].argmax()
            labels[best] = 1

        pos = labels == 1
        matched_gt[pos] = gt_boxes[argmax_iou[pos]]
        return labels, matched_gt

    def _rpn_losses(self, objectness, bbox_deltas, anchors, targets, images):
        device = objectness.device
        B = len(targets)
        total_obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_imgs = 0

        for b in range(B):
            gt_boxes = targets[b]["boxes"].to(device)
            labels, matched_gt = self._match_anchors(anchors, gt_boxes)

            obj_logits = objectness[b]
            obj_targets = labels.clone()
            obj_targets[labels == -1] = 0  # 忽略位置当作负（参与 BCE 但不进回归）

            pos = labels == 1
            neg = labels == 0
            n_pos = pos.sum().item()
            n_neg = neg.sum().item()

            # 采样 256 个（正负各最多 128）
            pos_idx = pos.nonzero().flatten()
            neg_idx = neg.nonzero().flatten()
            keep_pos = min(n_pos, 128)
            keep_neg = min(n_neg, 256 - keep_pos)
            if keep_pos > 0:
                pos_idx = pos_idx[torch.randperm(n_pos, device=device)[:keep_pos]]
            if keep_neg > 0:
                neg_idx = neg_idx[torch.randperm(n_neg, device=device)[:keep_neg]]
            sample_idx = torch.cat([pos_idx, neg_idx])

            if sample_idx.numel() == 0:
                continue

            obj_loss = F.binary_cross_entropy_with_logits(
                obj_logits[sample_idx], obj_targets[sample_idx].float())

            reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if pos_idx.numel() > 0:
                gt_deltas = _encode_boxes(matched_gt[pos_idx], anchors[pos_idx])
                reg_loss = F.smooth_l1_loss(bbox_deltas[b][pos_idx], gt_deltas, beta=1.0)

            total_obj_loss = total_obj_loss + obj_loss
            total_reg_loss = total_reg_loss + reg_loss
            n_imgs += 1

        if n_imgs == 0:
            return total_obj_loss, total_reg_loss
        return total_obj_loss / n_imgs, total_reg_loss / n_imgs

    def _proposals_from_rpn(self, feats, images, top_n=1000, iou_thresh=0.7):
        """推理/训练共享：由 RPN 生成 proposals。返回 list[Tensor[K_i,4]]。"""
        objectness, bbox_deltas = self._rpn_forward(feats)
        anchors, strides = self._anchors(feats, images)
        device = objectness.device
        B = objectness.shape[0]
        proposals = []
        for b in range(B):
            scores = objectness[b].sigmoid()
            keep = torch.topk(scores, min(top_n, scores.numel())).indices
            boxes = _decode_boxes(bbox_deltas[b][keep], anchors[keep])
            boxes = clip_boxes_to_image(boxes, images.shape[-2:])
            sel_scores = scores[keep]
            # remove_small_boxes 返回的是保留框的索引（torchvision 0.28）。
            small_keep = remove_small_boxes(boxes, min_size=2.0)
            boxes = boxes[small_keep]
            sel_scores = sel_scores[small_keep]
            keep2 = nms(boxes, sel_scores, iou_thresh) \
                if boxes.numel() > 0 else torch.zeros(0, dtype=torch.long, device=device)
            boxes = boxes[keep2][:top_n]
            proposals.append(boxes)
        return proposals

    # -- RoI 采样与损失 ---------------------------------------------------
    def _sample_rois(self, proposals, targets, pos_thresh, batch_size=512, pos_fraction=0.25):
        """采样正负 RoI，返回 (sampled_boxes, gt_boxes, labels) —— 展平后跨图。"""
        device = proposals[0].device
        boxes_list, gt_list, labels_list = [], [], []
        for b, prop in enumerate(proposals):
            gt_boxes = targets[b]["boxes"].to(device)
            if gt_boxes.numel() == 0 or prop.numel() == 0:
                # 无 GT：全部为负
                if prop.numel() > 0:
                    neg = prop
                    boxes_list.append(neg)
                    gt_list.append(neg)  # 占位（不进回归）
                    labels_list.append(torch.zeros(neg.shape[0], dtype=torch.long, device=device))
                continue
            iou = box_iou(prop, gt_boxes)
            max_iou, argmax = iou.max(dim=1)
            pos = max_iou >= pos_thresh
            neg = (max_iou >= 0.1) & (max_iou < pos_thresh)

            n_pos = min(int(batch_size * pos_fraction), pos.sum().item())
            n_neg = min(batch_size - n_pos, neg.sum().item())
            pos_idx = pos.nonzero().flatten()
            neg_idx = neg.nonzero().flatten()
            if n_pos > 0:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:n_pos]]
            if n_neg > 0:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:n_neg]]
            sel = torch.cat([pos_idx, neg_idx])

            if sel.numel() == 0:
                continue
            boxes_list.append(prop[sel])
            gt_list.append(gt_boxes[argmax[sel]])
            # 类别：正样本取真实类别（+1 因为背景=0），负样本=0
            lbls = torch.zeros(sel.numel(), dtype=torch.long, device=device)
            sel_pos = max_iou[sel] >= pos_thresh
            gt_lbl = targets[b]["labels"].to(device)
            lbls[sel_pos] = gt_lbl[argmax[sel][sel_pos]] + 1
            labels_list.append(lbls)

        return boxes_list, gt_list, labels_list

    def _roi_losses(self, feats, proposals, targets, images, pos_thresh=0.5, roi_head=None):
        device = proposals[0].device
        roi_head = roi_head if roi_head is not None else self.roi_head
        boxes_list, gt_list, labels_list = self._sample_rois(
            proposals, targets, pos_thresh)

        if len(boxes_list) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero

        boxes_cat = torch.cat(boxes_list)
        labels_cat = torch.cat(labels_list)
        gt_cat = torch.cat(gt_list)

        pooled = self.roi_pool(
            self._fpn_ordered(feats), boxes_list, [images.shape[-2:]] * len(boxes_list))
        cls_logits, bbox_deltas = roi_head(pooled)

        cls_loss = F.cross_entropy(cls_logits, labels_cat)
        pos_mask = labels_cat > 0
        reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if pos_mask.sum() > 0:
            gt_deltas = _encode_boxes(gt_cat[pos_mask], boxes_cat[pos_mask])
            reg_loss = F.smooth_l1_loss(bbox_deltas[pos_mask], gt_deltas, beta=1.0)
        return cls_loss, reg_loss

    # -- 前向 -------------------------------------------------------------
    def forward(self, features, images, targets=None):
        feats = self._fpn_features(features)
        if targets is not None:
            return self._forward_train(feats, images, targets)
        return self._forward_inference(feats, images)

    def get_lesion_features(self, features, images):
        feats = self._fpn_features(features)
        return F.adaptive_avg_pool2d(feats[-1], (1, 1)).flatten(1)


# ---------------------------------------------------------------------------
# Faster R-CNN
# ---------------------------------------------------------------------------

class FasterRCNNHead(_TwoStageHead):
    """标准两阶段 Faster R-CNN 头。"""

    def __init__(self, in_channels_list, fpn_channels, num_classes):
        super().__init__(in_channels_list, fpn_channels, num_classes)
        self.roi_head = RoIHead(fpn_channels, num_classes)

    def _forward_train(self, feats, images, targets):
        objectness, bbox_deltas = self._rpn_forward(feats)
        anchors, _ = self._anchors(feats, images)
        loss_rpn_obj, loss_rpn_reg = self._rpn_losses(
            objectness, bbox_deltas, anchors, targets, images)

        proposals = self._proposals_from_rpn(feats, images)
        loss_roi_cls, loss_roi_reg = self._roi_losses(
            feats, proposals, targets, images, pos_thresh=0.5)

        return {"losses": {
            "loss_det_rpn_cls": loss_rpn_obj,
            "loss_det_rpn_reg": loss_rpn_reg,
            "loss_det_roi_cls": loss_roi_cls,
            "loss_det_roi_reg": loss_roi_reg,
        }}

    def _forward_inference(self, feats, images):
        device = images.device
        B = images.shape[0]
        proposals = self._proposals_from_rpn(feats, images, top_n=1000)

        all_boxes, all_scores, all_labels = [], [], []
        for b in range(B):
            if proposals[b].numel() == 0:
                all_boxes.append(torch.zeros((0, 4), device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            pooled = self.roi_pool(
                self._fpn_ordered(feats), [proposals[b]],
                [images.shape[-2:]])
            cls_logits, bbox_deltas = self.roi_head(pooled)
            probs = cls_logits.softmax(dim=1)  # [K, C+1]
            boxes = _decode_boxes(bbox_deltas, proposals[b])
            boxes = clip_boxes_to_image(boxes, images.shape[-2:])

            keep_boxes, keep_scores, keep_labels = [], [], []
            for cls in range(1, self.num_classes + 1):
                scores = probs[:, cls]
                idx = torch.topk(scores, min(scores.numel(), 500)).indices
                if idx.numel() == 0:
                    continue
                sel_boxes = boxes[idx]
                sel_scores = scores[idx]
                keep = nms(sel_boxes, sel_scores, 0.5) if sel_boxes.numel() else \
                    torch.zeros(0, dtype=torch.long, device=device)
                if keep.numel() == 0:
                    continue
                keep_boxes.append(sel_boxes[keep])
                keep_scores.append(sel_scores[keep])
                keep_labels.append(torch.full_like(keep, cls - 1))

            if keep_boxes:
                boxes_b = torch.cat(keep_boxes)
                scores_b = torch.cat(keep_scores)
                labels_b = torch.cat(keep_labels)
                # 全局 top-k 截断
                topk = min(100, scores_b.numel())
                top = torch.topk(scores_b, topk).indices
                all_boxes.append(boxes_b[top])
                all_scores.append(scores_b[top])
                all_labels.append(labels_b[top])
            else:
                all_boxes.append(torch.zeros((0, 4), device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))

        return {"boxes": all_boxes, "labels": all_labels, "scores": all_scores}


# ---------------------------------------------------------------------------
# Cascade R-CNN
# ---------------------------------------------------------------------------

class CascadeRCNNHead(_TwoStageHead):
    """3 阶段级联 RoI 头（IoU 阈值 0.5 / 0.6 / 0.7）。"""

    CASCADE_THRESH = [0.5, 0.6, 0.7]

    def __init__(self, in_channels_list, fpn_channels, num_classes):
        super().__init__(in_channels_list, fpn_channels, num_classes)
        self.roi_heads = nn.ModuleList(
            [RoIHead(fpn_channels, num_classes) for _ in self.CASCADE_THRESH])

    def _forward_train(self, feats, images, targets):
        objectness, bbox_deltas = self._rpn_forward(feats)
        anchors, _ = self._anchors(feats, images)
        loss_rpn_obj, loss_rpn_reg = self._rpn_losses(
            objectness, bbox_deltas, anchors, targets, images)

        proposals = self._proposals_from_rpn(feats, images)
        total_cls = torch.tensor(0.0, device=images.device, requires_grad=True)
        total_reg = torch.tensor(0.0, device=images.device, requires_grad=True)

        for stage, thresh in enumerate(self.CASCADE_THRESH):
            cls_loss, reg_loss = self._roi_losses(
                feats, proposals, targets, images, pos_thresh=thresh,
                roi_head=self.roi_heads[stage])
            total_cls = total_cls + cls_loss
            total_reg = total_reg + reg_loss
            # 用当前阶段精修 proposals 供下一阶段
            if stage < len(self.CASCADE_THRESH) - 1:
                proposals = self._refine_proposals(feats, proposals, images, stage)

        return {"losses": {
            "loss_det_rpn_cls": loss_rpn_obj,
            "loss_det_rpn_reg": loss_rpn_reg,
            "loss_det_roi_cls": total_cls,
            "loss_det_roi_reg": total_reg,
        }}

    def _refine_proposals(self, feats, proposals, images, stage):
        """用第 stage 个 RoI 头精修 proposals。"""
        new_proposals = []
        roi_head = self.roi_heads[stage]
        for b, prop in enumerate(proposals):
            if prop.numel() == 0:
                new_proposals.append(prop)
                continue
            pooled = self.roi_pool(
                self._fpn_ordered(feats), [prop], [images.shape[-2:]])
            _, bbox_deltas = roi_head(pooled)
            boxes = _decode_boxes(bbox_deltas, prop)
            boxes = clip_boxes_to_image(boxes, images.shape[-2:])
            new_proposals.append(boxes)
        return new_proposals

    def _forward_inference(self, feats, images):
        device = images.device
        B = images.shape[0]
        proposals = self._proposals_from_rpn(feats, images, top_n=1000)

        # 依次用各阶段精修
        for stage in range(len(self.CASCADE_THRESH) - 1):
            proposals = self._refine_proposals(feats, proposals, images, stage)

        roi_head = self.roi_heads[-1]
        all_boxes, all_scores, all_labels = [], [], []
        for b in range(B):
            if proposals[b].numel() == 0:
                all_boxes.append(torch.zeros((0, 4), device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            pooled = self.roi_pool(
                self._fpn_ordered(feats), [proposals[b]], [images.shape[-2:]])
            cls_logits, bbox_deltas = roi_head(pooled)
            probs = cls_logits.softmax(dim=1)
            boxes = _decode_boxes(bbox_deltas, proposals[b])
            boxes = clip_boxes_to_image(boxes, images.shape[-2:])

            keep_boxes, keep_scores, keep_labels = [], [], []
            for cls in range(1, self.num_classes + 1):
                scores = probs[:, cls]
                idx = torch.topk(scores, min(scores.numel(), 500)).indices
                sel_boxes, sel_scores = boxes[idx], scores[idx]
                keep = nms(sel_boxes, sel_scores, 0.5) if sel_boxes.numel() else \
                    torch.zeros(0, dtype=torch.long, device=device)
                if keep.numel() == 0:
                    continue
                keep_boxes.append(sel_boxes[keep])
                keep_scores.append(sel_scores[keep])
                keep_labels.append(torch.full_like(keep, cls - 1))

            if keep_boxes:
                boxes_b = torch.cat(keep_boxes)
                scores_b = torch.cat(keep_scores)
                labels_b = torch.cat(keep_labels)
                topk = min(100, scores_b.numel())
                top = torch.topk(scores_b, topk).indices
                all_boxes.append(boxes_b[top])
                all_scores.append(scores_b[top])
                all_labels.append(labels_b[top])
            else:
                all_boxes.append(torch.zeros((0, 4), device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))

        return {"boxes": all_boxes, "labels": all_labels, "scores": all_scores}
