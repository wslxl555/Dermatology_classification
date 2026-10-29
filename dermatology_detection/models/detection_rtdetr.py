"""
RT-DETR 风格（DETR）检测头：transformer encoder-decoder + 集合预测。

简化实现（自包含纯 PyTorch）：
  - 多尺度 FPN 特征 → 1x1 投影到 d_model → 展平 + 2D 正弦位置编码 + 层级嵌入。
  - Transformer encoder（自注意力）→ 记忆张量。
  - Transformer decoder + 可学习 object queries → 输出序列。
  - 分类头（num_classes + 背景）+ 框头（归一化 cxcywh）。
  - 训练用匈牙利/贪心一对一匹配，损失 = CE 分类 + L1 + GIoU。

与 FCOS / Faster R-CNN 共享统一接口。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork, batched_nms


def _pos_encoding_2d(d_model: int, h: int, w: int, device) -> torch.Tensor:
    """2D 正弦位置编码 → [d_model, h, w]（channels-first）。

    前 d_model//2 通道编码 Y 坐标，后 d_model//2 通道编码 X 坐标，
    相邻通道交替 sin/cos。
    """
    d = d_model // 2
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    div = 10000.0 ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)  # [d/2]

    def _encode(coords: torch.Tensor) -> torch.Tensor:
        """coords [N] → [N, d]（sin/cos 按通道交织）。"""
        sin = torch.sin(coords[:, None] / div[None, :])   # [N, d/2]
        cos = torch.cos(coords[:, None] / div[None, :])   # [N, d/2]
        return torch.stack([sin, cos], dim=-1).reshape(-1, d)  # [N, d]

    y_enc = _encode(ys).transpose(0, 1)  # [d, h]
    x_enc = _encode(xs).transpose(0, 1)  # [d, w]
    pe = torch.zeros(d_model, h, w, device=device)
    pe[:d] = y_enc.unsqueeze(-1)   # [d, h, 1] → 广播到 [d, h, w]
    pe[d:] = x_enc.unsqueeze(-2)   # [d, 1, w] → 广播到 [d, h, w]
    return pe


def _giou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred/gt: [N, 4] (cxcywh 或 xyxy 均可，先转 xyxy)。"""
    px1 = pred[:, 0] - pred[:, 2] / 2
    py1 = pred[:, 1] - pred[:, 3] / 2
    px2 = pred[:, 0] + pred[:, 2] / 2
    py2 = pred[:, 1] + pred[:, 3] / 2
    gx1 = gt[:, 0] - gt[:, 2] / 2
    gy1 = gt[:, 1] - gt[:, 3] / 2
    gx2 = gt[:, 0] + gt[:, 2] / 2
    gy2 = gt[:, 1] + gt[:, 3] / 2

    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    gw = (gx2 - gx1).clamp(min=0)
    gh = (gy2 - gy1).clamp(min=0)

    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    iou = inter / union.clamp(min=1e-6)

    cx1 = torch.min(px1, gx1)
    cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2)
    cy2 = torch.max(py2, gy2)
    cw = (cx2 - cx1).clamp(min=1e-6)
    ch = (cy2 - cy1).clamp(min=1e-6)
    carea = cw * ch
    return 1 - iou + (carea - union) / carea.clamp(min=1e-6)


class RTDETRHead(nn.Module):
    """RT-DETR 风格 transformer 检测头。"""

    def __init__(self, in_channels_list, fpn_channels, num_classes,
                 d_model: int = 256, nhead: int = 8, num_queries: int = 100,
                 num_layers: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_levels = len(in_channels_list)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=fpn_channels,
        )
        self.input_proj = nn.ModuleList(
            [nn.Conv2d(fpn_channels, d_model, 1) for _ in range(self.num_levels)])
        self.level_embed = nn.Parameter(
            torch.zeros(self.num_levels, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.box_head = nn.Linear(d_model, 4)
        nn.init.constant_(self.box_head.bias, 0.0)

    def _fpn_features(self, features):
        fpn_out = self.fpn(features)
        keys = sorted(fpn_out.keys())
        return [fpn_out[k] for k in keys]

    def _build_src(self, feats, images):
        B = images.shape[0]
        srcs = []
        for lvl, feat in enumerate(feats):
            B, C, H, W = feat.shape
            proj = self.input_proj[lvl](feat)          # [B, d, H, W]
            pe = _pos_encoding_2d(self.d_model, H, W, feat.device)
            proj = proj + pe.unsqueeze(0) + self.level_embed[lvl].view(1, -1, 1, 1)
            srcs.append(proj.flatten(2).transpose(1, 2))  # [B, H*W, d]
        return torch.cat(srcs, dim=1)  # [B, N, d]

    def forward(self, features, images, targets=None):
        feats = self._fpn_features(features)
        src = self._build_src(feats, images)
        memory = self.encoder(src)
        B = images.shape[0]
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.decoder(tgt, memory)                    # [B, Q, d]
        cls_logits = self.class_head(hs)                  # [B, Q, C+1]
        boxes = self.box_head(hs).sigmoid()               # [B, Q, 4] cxcywh 归一化

        if targets is not None:
            return self._forward_train(cls_logits, boxes, targets, images)
        return self._forward_inference(cls_logits, boxes, images)

    def get_lesion_features(self, features, images):
        feats = self._fpn_features(features)
        return F.adaptive_avg_pool2d(feats[-1], (1, 1)).flatten(1)

    # -- 匹配 -------------------------------------------------------------
    def _greedy_match(self, cls_probs, gt_labels, gt_boxes):
        """贪心一对一匹配。返回 (pred_idx, gt_idx) 两个张量。"""
        device = cls_probs.device
        Q = cls_probs.shape[0]
        used = torch.zeros(Q, dtype=torch.bool, device=device)
        pred_idx, gt_idx = [], []
        # 每个 GT 选择未被占用的最高分类分预测
        for m, lbl in enumerate(gt_labels):
            scores = cls_probs[:, int(lbl) + 1].clone()  # [Q]
            scores[used] = -1.0
            q = int(scores.argmax())
            if scores[q].item() <= 0.0:
                continue
            used[q] = True
            pred_idx.append(q)
            gt_idx.append(m)
        return (torch.tensor(pred_idx, device=device, dtype=torch.long),
                torch.tensor(gt_idx, device=device, dtype=torch.long))

    def _forward_train(self, cls_logits, boxes, targets, images):
        device = cls_logits.device
        B = cls_logits.shape[0]
        total_cls = torch.tensor(0.0, device=device, requires_grad=True)
        total_box = torch.tensor(0.0, device=device, requires_grad=True)
        total_giou = torch.tensor(0.0, device=device, requires_grad=True)
        n = 0

        for b in range(B):
            gt_boxes = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device).long()
            if gt_boxes.numel() == 0:
                continue
            H, W = images.shape[-2:]
            # GT 转 cxcywh 归一化
            gt_cxcywh = torch.stack([
                (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 / W,
                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 / H,
                (gt_boxes[:, 2] - gt_boxes[:, 0]) / W,
                (gt_boxes[:, 3] - gt_boxes[:, 1]) / H,
            ], dim=1)

            cls_probs = cls_logits[b].softmax(dim=1)
            pred_idx, gt_idx = self._greedy_match(cls_probs, gt_labels, gt_boxes)

            if pred_idx.numel() == 0:
                continue

            cls_target = torch.zeros(pred_idx.numel(), dtype=torch.long, device=device)
            cls_target[:] = gt_labels[gt_idx] + 1  # 背景=0
            cls_loss = F.cross_entropy(cls_logits[b][pred_idx], cls_target)

            pred_box = boxes[b][pred_idx]
            gt_box = gt_cxcywh[gt_idx]
            box_loss = F.l1_loss(pred_box, gt_box)
            giou_loss = _giou_loss(pred_box, gt_box).mean()

            total_cls = total_cls + cls_loss
            total_box = total_box + box_loss
            total_giou = total_giou + giou_loss
            n += 1

        if n > 0:
            total_cls = total_cls / n
            total_box = total_box / n
            total_giou = total_giou / n

        return {"losses": {
            "loss_det_cls": total_cls,
            "loss_det_reg": total_box,
            "loss_det_giou": total_giou,
        }}

    def _forward_inference(self, cls_logits, boxes, images):
        device = images.device
        B = images.shape[0]
        H, W = images.shape[-2:]
        probs = cls_logits.softmax(dim=-1)  # [B, Q, C+1]

        out_boxes, out_scores, out_labels = [], [], []
        for b in range(B):
            obj = 1 - probs[b, :, 0]  # 非背景
            pb = boxes[b]             # cxcywh 归一化
            abs_boxes = torch.stack([
                (pb[:, 0] - pb[:, 2] / 2) * W,
                (pb[:, 1] - pb[:, 3] / 2) * H,
                (pb[:, 0] + pb[:, 2] / 2) * W,
                (pb[:, 1] + pb[:, 3] / 2) * H,
            ], dim=1)

            keep_boxes, keep_scores, keep_labels = [], [], []
            for cls in range(1, self.num_classes + 1):
                scores = probs[b, :, cls]
                mask = scores > 0.25
                if mask.sum() == 0:
                    continue
                sel_boxes = abs_boxes[mask]
                sel_scores = scores[mask]
                keep = batched_nms(sel_boxes, sel_scores,
                                   torch.full_like(mask[mask].long(), cls - 1), 0.5)
                if keep.numel() == 0:
                    continue
                keep_boxes.append(sel_boxes[keep])
                keep_scores.append(sel_scores[keep])
                keep_labels.append(torch.full_like(keep, cls - 1))

            if keep_boxes:
                bb = torch.cat(keep_boxes)
                ss = torch.cat(keep_scores)
                ll = torch.cat(keep_labels)
                topk = min(100, ss.numel())
                top = torch.topk(ss, topk).indices
                out_boxes.append(bb[top])
                out_scores.append(ss[top])
                out_labels.append(ll[top])
            else:
                out_boxes.append(torch.zeros((0, 4), device=device))
                out_scores.append(torch.zeros(0, device=device))
                out_labels.append(torch.zeros(0, dtype=torch.long, device=device))

        return {"boxes": out_boxes, "labels": out_labels, "scores": out_scores}
