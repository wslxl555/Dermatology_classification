"""
YOLOv8 风格解耦 anchor-free 检测头（自包含纯 PyTorch 实现）。

每个 FPN 层级使用解耦头：
  - cls 分支：2×3x3 conv + 1x1 conv → num_classes（sigmoid，无背景类）
  - reg 分支：2×3x3 conv + 1x1 conv → 4×reg_max（Distribution Focal Loss）

损失：BCE 分类 + CIoU 回归 + DFL。分配：中心落在 GT 框内的位置为正样本。
与 FCOS / Faster R-CNN 共享统一接口。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork, batched_nms


class _YOLOLevel(nn.Module):
    """单个 FPN 层级的解耦检测头。"""

    def __init__(self, in_channels: int, num_classes: int, reg_max: int):
        super().__init__()
        c = in_channels
        self.cls_convs = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.SiLU(inplace=True),
        )
        self.reg_convs = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(c, num_classes, 1)
        self.reg_pred = nn.Conv2d(c, 4 * reg_max, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        return self.cls_pred(self.cls_convs(x)), self.reg_pred(self.reg_convs(x))


class YOLOv8Head(nn.Module):
    """YOLOv8 风格解耦 anchor-free 检测头。"""

    def __init__(self, in_channels_list, fpn_channels, num_classes, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_levels = len(in_channels_list)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=fpn_channels,
        )
        self.levels = nn.ModuleList(
            [_YOLOLevel(fpn_channels, num_classes, reg_max)
             for _ in range(self.num_levels)]
        )
        self.proj = torch.arange(reg_max, dtype=torch.float32)
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    # -- 特征组织 ---------------------------------------------------------
    def _fpn_features(self, features):
        fpn_out = self.fpn(features)
        keys = sorted(fpn_out.keys())
        return [fpn_out[k] for k in keys]

    def _strides(self, feats, images):
        img_h = images.shape[-2]
        return [img_h // f.shape[-2] for f in feats]

    def forward(self, features, images, targets=None):
        feats = self._fpn_features(features)
        if targets is not None:
            return self._forward_train(feats, images, targets)
        return self._forward_inference(feats, images)

    def get_lesion_features(self, features, images):
        feats = self._fpn_features(features)
        return F.adaptive_avg_pool2d(feats[-1], (1, 1)).flatten(1)

    # -- 预测 -------------------------------------------------------------
    def _predict(self, feats):
        cls_preds, reg_preds = [], []
        for lvl, feat in zip(self.levels, feats):
            cls, reg = lvl(feat)
            cls_preds.append(cls)   # [B, C, H, W]
            reg_preds.append(reg)   # [B, 4*reg_max, H, W]
        return cls_preds, reg_preds

    def _decode_dist(self, reg_pred):
        """[*, 4*reg_max, H, W] → 距离 [*, 4, H, W]（支持带/不带 batch 维）。"""
        proj = self.proj.to(reg_pred.device)
        if reg_pred.dim() == 4:  # [B, 4*reg_max, H, W]
            B, _, H, W = reg_pred.shape
            reg = reg_pred.view(B, 4, self.reg_max, H, W).softmax(dim=2)
            return (reg * proj.view(1, 1, -1, 1, 1)).sum(dim=2)
        else:                    # [4*reg_max, H, W]
            _, H, W = reg_pred.shape
            reg = reg_pred.view(4, self.reg_max, H, W).softmax(dim=1)
            return (reg * proj.view(1, -1, 1, 1)).sum(dim=1)

    # -- 训练 -------------------------------------------------------------
    def _forward_train(self, feats, images, targets):
        device = images.device
        cls_preds, reg_preds = self._predict(feats)
        strides = self._strides(feats, images)

        total_cls = torch.tensor(0.0, device=device, requires_grad=True)
        total_reg = torch.tensor(0.0, device=device, requires_grad=True)
        total_dfl = torch.tensor(0.0, device=device, requires_grad=True)
        n_pos_total = 0

        for b, target in enumerate(targets):
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            if gt_boxes.numel() == 0:
                continue
            for gi, (box, lbl) in enumerate(zip(gt_boxes, gt_labels)):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                level = int(math.floor(math.log2(math.sqrt(max(area, 1.0))) - 2))
                level = max(0, min(level, self.num_levels - 1))

                cls = cls_preds[level][b]     # [C, H, W]
                reg = reg_preds[level][b]     # [4*reg_max, H, W]
                stride = strides[level]
                H, W = cls.shape[-2:]

                ys = (torch.arange(H, device=device).float() + 0.5) * stride
                xs = (torch.arange(W, device=device).float() + 0.5) * stride
                in_x = (xs >= x1) & (xs <= x2)
                in_y = (ys >= y1) & (ys <= y2)
                mask = in_y.unsqueeze(1) & in_x.unsqueeze(0)
                if mask.sum() == 0:
                    continue

                # 中心最接近 GT 中心的位置作为主要正样本
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                gx = (torch.arange(W, device=device).float() + 0.5) * stride
                gy = (torch.arange(H, device=device).float() + 0.5) * stride
                dist = (gy.unsqueeze(1) - cy) ** 2 + (gx.unsqueeze(0) - cx) ** 2
                dist[~mask] = float("inf")
                py, px = torch.where(dist == dist.min())
                py, px = py[0], px[0]

                # 分类：正样本 BCE（sigmoid focal）
                target_onehot = torch.zeros(self.num_classes, device=device)
                target_onehot[lbl.long()] = 1.0
                cls_logit = cls[:, py, px]
                cls_loss = _bce_focal(cls_logit, target_onehot, self.focal_alpha, self.focal_gamma)

                # 距离分布解码
                dist_map = self._decode_dist(reg)          # [4, H, W]
                pred_l, pred_t, pred_r, pred_b = (
                    dist_map[0, py, px], dist_map[1, py, px],
                    dist_map[2, py, px], dist_map[3, py, px])
                gt_l = (cx - x1).clamp(min=0)
                gt_t = (cy - y1).clamp(min=0)
                gt_r = (x2 - cx).clamp(min=0)
                gt_b = (y2 - cy).clamp(min=0)

                # CIoU 回归损失（用预测与 GT 的完整框）
                px1, py1 = cx - pred_l, cy - pred_t
                px2, py2 = cx + pred_r, cy + pred_b
                iou_loss = _ciou_loss(
                    torch.stack([px1, py1, px2, py2]),
                    torch.stack([x1, y1, x2, y2]))

                # DFL：对四条边各自的分布
                dfl = _dfl_loss(reg[:, py, px], gt_l, gt_t, gt_r, gt_b, self.reg_max)

                total_cls = total_cls + cls_loss
                total_reg = total_reg + iou_loss
                total_dfl = total_dfl + dfl
                n_pos_total += 1

        if n_pos_total > 0:
            total_cls = total_cls / n_pos_total
            total_reg = total_reg / n_pos_total
            total_dfl = total_dfl / n_pos_total

        return {"losses": {
            "loss_det_cls": total_cls,
            "loss_det_reg": total_reg,
            "loss_det_dfl": total_dfl,
        }}

    # -- 推理 -------------------------------------------------------------
    def _forward_inference(self, feats, images):
        device = images.device
        B = images.shape[0]
        H_in, W_in = images.shape[-2:]
        cls_preds, reg_preds = self._predict(feats)
        strides = self._strides(feats, images)

        all_boxes = [[] for _ in range(B)]
        all_scores = [[] for _ in range(B)]
        all_labels = [[] for _ in range(B)]

        for level in range(self.num_levels):
            cls = cls_preds[level].sigmoid()          # [B, C, H, W]
            dist = self._decode_dist(reg_preds[level])  # [B, 4, H, W]
            stride = strides[level]
            H, W = cls.shape[-2:]
            ys = (torch.arange(H, device=device).float() + 0.5) * stride
            xs = (torch.arange(W, device=device).float() + 0.5) * stride
            cx = xs.view(1, 1, W).expand(B, H, W)
            cy = ys.view(1, H, 1).expand(B, H, W)

            x1 = cx - dist[:, 0]
            y1 = cy - dist[:, 1]
            x2 = cx + dist[:, 2]
            y2 = cy + dist[:, 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, H, W, 4]

            for b in range(B):
                scores_b, labels_b = cls[b].max(dim=0)     # [H, W]
                mask = scores_b > 0.25
                if mask.sum() == 0:
                    continue
                sel_boxes = boxes[b][mask]
                sel_scores = scores_b[mask]
                sel_labels = labels_b[mask]
                keep = batched_nms(sel_boxes, sel_scores, sel_labels, 0.5)
                all_boxes[b].append(sel_boxes[keep])
                all_scores[b].append(sel_scores[keep])
                all_labels[b].append(sel_labels[keep])

        out_boxes, out_scores, out_labels = [], [], []
        for b in range(B):
            if all_boxes[b]:
                bb = torch.cat(all_boxes[b])
                ss = torch.cat(all_scores[b])
                ll = torch.cat(all_labels[b])
                topk = min(100, ss.numel())
                top = torch.topk(ss, topk).indices
                bb = bb[top].clamp(min=0)
                bb[:, 0::2] = bb[:, 0::2].clamp(max=W_in)
                bb[:, 1::2] = bb[:, 1::2].clamp(max=H_in)
                out_boxes.append(bb)
                out_scores.append(ss[top])
                out_labels.append(ll[top])
            else:
                out_boxes.append(torch.zeros((0, 4), device=device))
                out_scores.append(torch.zeros(0, device=device))
                out_labels.append(torch.zeros(0, dtype=torch.long, device=device))

        return {"boxes": out_boxes, "labels": out_labels, "scores": out_scores}


# ---------------------------------------------------------------------------
# 损失辅助函数
# ---------------------------------------------------------------------------

def _bce_focal(logits, targets, alpha, gamma):
    prob = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    ce = -targets * torch.log(prob) - (1 - targets) * torch.log(1 - prob)
    p_t = targets * prob + (1 - targets) * (1 - prob)
    loss = alpha * (1 - p_t) ** gamma * ce
    return loss.sum()


def _ciou_loss(pred, gt):
    px1, py1, px2, py2 = pred[0], pred[1], pred[2], pred[3]
    gx1, gy1, gx2, gy2 = gt[0], gt[1], gt[2], gt[3]
    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    gw = (gx2 - gx1).clamp(min=0)
    gh = (gy2 - gy1).clamp(min=0)

    ix1, iy1 = torch.max(px1, gx1), torch.max(py1, gy1)
    ix2, iy2 = torch.min(px2, gx2), torch.min(py2, gy2)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    iou = inter / union.clamp(min=1e-6)

    # 外接框
    cx1, cy1 = torch.min(px1, gx1), torch.min(py1, gy1)
    cx2, cy2 = torch.max(px2, gx2), torch.max(py2, gy2)
    cw = (cx2 - cx1).clamp(min=1e-6)
    ch = (cy2 - cy1).clamp(min=1e-6)

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    v = (4 / (math.pi ** 2)) * (
        torch.atan(gw / gh.clamp(min=1e-6)) - torch.atan(pw / ph.clamp(min=1e-6))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=1e-6)
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    c2 = cw ** 2 + ch ** 2
    return 1 - iou + rho2 / c2.clamp(min=1e-6) + alpha * v


def _dfl_loss(reg, gt_l, gt_t, gt_r, gt_b, reg_max):
    """reg: [4*reg_max]（四条边连续排列），计算每边的 DFL。"""
    loss = torch.tensor(0.0, device=reg.device)
    gts = [gt_l, gt_t, gt_r, gt_b]
    for i, gt in enumerate(gts):
        seg = reg[i * reg_max:(i + 1) * reg_max]  # [reg_max]
        gt = gt.clamp(0, reg_max - 1)
        lo = gt.floor().long()
        hi = (lo + 1).clamp(max=reg_max - 1)
        wi = hi - gt
        prob = seg.softmax(dim=0)
        loss = loss - (wi * torch.log(prob[lo].clamp(min=1e-6))
                       + (1 - wi) * torch.log(prob[hi].clamp(min=1e-6)))
    return loss / 4.0
