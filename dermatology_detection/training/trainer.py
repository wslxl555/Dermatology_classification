"""
多任务湿疹分类模型的训练循环。
支持混合精度、梯度累积、早停与 TensorBoard 日志记录。
"""

import os
import time
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from training.losses import MultiTaskLoss, compute_class_weights
from training.metrics import MetricsTracker
from config import (
    DISEASE_CLASSES, DISEASE_CLASSES_ZH,
    TRAINING_CONFIG, CHECKPOINT_DIR, LOG_DIR,
)


class Trainer:
    """MultiTaskEczemaModel 的训练器。"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict = None,
        device: str = None,
        train_eval_loader: DataLoader = None,
        train_class_counts: list[int] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_eval_loader = train_eval_loader
        self.config = config or TRAINING_CONFIG
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)

        # 类别权重必须用“真实训练分布”计算，不能用带 WeightedRandomSampler 的
        # loader（否则会被采样成均衡分布，alpha 被中和成 ~[1,1,1]）。
        # 优先使用调用方传入的真实每类样本数。
        if train_class_counts is not None:
            train_counts = [int(c) for c in train_class_counts]
        else:
            train_counts = self._count_samples_per_class(train_loader)
        class_weights = compute_class_weights(train_counts)

        # 损失函数
        self.criterion = MultiTaskLoss(
            num_disease_classes=len(DISEASE_CLASSES),
            class_weights=class_weights,
            focal_gamma=self.config["focal_gamma"],
            label_smoothing=self.config["label_smoothing"],
            loss_disease_weight=self.config.get("loss_disease_weight", 1.0),
            loss_global_weight=self.config.get("loss_global_weight", 0.3),
            loss_detection_weight=self.config.get("loss_detection_weight", 1.0),
            num_position_classes=self.config.get("num_position_classes", 4),
            loss_position_weight=self.config.get("loss_position_weight", 0.3),
        )

        # 按皮损类别频率自动计算检测分支的 Focal alpha / CE 权重(长尾缓解)
        self._apply_lesion_class_weights(train_loader)

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # 学习率调度:warmup + 单调余弦下降
        warmup = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            total_iters=self.config["warmup_epochs"] * len(train_loader),
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=(self.config["max_epochs"] - self.config["warmup_epochs"]) * len(train_loader),
            eta_min=self.config["min_lr"],
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config["warmup_epochs"] * len(train_loader)],
        )

        # 混合精度
        self.scaler = GradScaler("cuda", enabled=self.config["use_amp"] and "cuda" in self.device)

        # 训练状态跟踪
        self.current_epoch = 0
        self.start_epoch = 0   # fit() 从此 epoch 开始；续训时设为 checkpoint epoch + 1
        self.best_val_f1 = 0.0
        self.best_det_map = 0.0
        self.best_combined_score = -1.0  # 疾病 F1 + 检测 mAP 的联合选优分数
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.train_metrics = MetricsTracker(
            num_classes=len(DISEASE_CLASSES),
            class_names=DISEASE_CLASSES,
        )
        self.val_metrics = MetricsTracker(
            num_classes=len(DISEASE_CLASSES),
            class_names=DISEASE_CLASSES,
        )

        # TensorBoard 日志
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(LOG_DIR))
        except ImportError:
            self.writer = None

    def _count_samples_per_class(self, loader: DataLoader) -> list[int]:
        """统计 DataLoader 中每个疾病类别的样本数量。"""
        counts = [0] * len(DISEASE_CLASSES)
        for batch in loader:
            for label in batch["disease_label"]:
                counts[label.item()] += 1
        return counts

    def _count_lesion_boxes_per_class(self, loader: DataLoader) -> list[int]:
        """统计 DataLoader 中每个皮损类别的框数量(含复合标签展开后的重复框)。"""
        from config import NUM_LESION_CLASSES
        counts = [0] * NUM_LESION_CLASSES
        for batch in loader:
            labels_list = batch.get("lesion_labels")
            if labels_list is None:
                continue
            for labels in labels_list:
                for lbl in labels.tolist():
                    if 0 <= lbl < NUM_LESION_CLASSES:
                        counts[lbl] += 1
        return counts

    def _apply_lesion_class_weights(self, loader: DataLoader) -> None:
        """按皮损类别框数自动计算 Focal alpha / 分割 CE 权重,缓解长尾分布。

        逆 sqrt 频率再归一化到 (0, 1]:稀有类权重接近 1、高频类权重较小。
        FCOS/YOLOv8 头通过 `focal_alpha` 生效;分割式头通过 `cls_weights` 生效。
        """
        import numpy as np
        from config import NUM_LESION_CLASSES
        counts = self._count_lesion_boxes_per_class(loader)
        if len(counts) != NUM_LESION_CLASSES or sum(counts) == 0:
            return

        counts = np.array(counts, dtype=np.float32)
        counts = np.maximum(counts, 1.0)
        alpha = 1.0 / np.sqrt(counts)
        alpha = alpha / alpha.max()
        alpha_t = torch.tensor(alpha, dtype=torch.float32, device=self.device)

        head = getattr(self.model, "detection_head", None)
        if head is None:
            return
        if hasattr(head, "focal_alpha"):
            head.focal_alpha = alpha_t          # FCOS / YOLOv8 的 Focal alpha
        if hasattr(head, "cls_weights"):
            # 分割式检测头:背景 0.1 + 前景按类别加权
            head.cls_weights = torch.cat([
                torch.tensor([0.1], dtype=torch.float32, device=self.device),
                alpha_t,
            ])

    def train_epoch(self) -> dict:
        """执行一个训练 epoch。"""
        self.model.train()
        self.train_metrics.reset()
        epoch_losses = defaultdict(float)
        self._train_pos_correct = 0
        self._train_pos_total = 0

        start_time = time.time()
        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            disease_labels = batch["disease_label"].to(self.device)
            position_labels = batch["position_label"].to(self.device)

            # 如启用检测,则准备检测目标
            targets = None
            has_det_targets = False
            if self.config.get("detection_enabled", False):
                targets, has_det_targets = self._prepare_detection_targets(batch)

            with autocast("cuda", enabled=self.config["use_amp"]):
                output = self.model(images, targets=targets)

                # 对没有 bbox 标注的 batch,跳过检测损失
                det_losses = output.get("detection", {}).get("losses")
                if not has_det_targets:
                    det_losses = None

                total_loss, loss_dict = self.criterion(
                    disease_logits=output["disease_logits"],
                    global_logits=output["global_logits"],
                    disease_targets=disease_labels,
                    detection_losses=det_losses,
                    position_logits=output.get("position_logits"),
                    position_targets=position_labels,
                )

            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["gradient_clip_norm"],
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # 记录损失
            for k, v in loss_dict.items():
                epoch_losses[k] += v

            # 更新指标
            with torch.no_grad():
                self.train_metrics.update(
                    output["disease_logits"].detach().cpu().numpy(),
                    disease_labels.cpu().numpy(),
                )
                # 跟踪位置准确率
                if "position_logits" in output:
                    pos_logits = output["position_logits"].detach().cpu()
                    pos_targets = position_labels.cpu()
                    valid_mask = pos_targets >= 0
                    if valid_mask.sum() > 0:
                        pos_pred = pos_logits[valid_mask].argmax(dim=-1)
                        pos_correct = (pos_pred == pos_targets[valid_mask]).sum().item()
                        self._train_pos_correct += pos_correct
                        self._train_pos_total += valid_mask.sum().item()

            if step % self.config["log_interval"] == 0 and self.writer:
                global_step = self.current_epoch * len(self.train_loader) + step
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f"train/{k}", v, global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], global_step)

        # epoch 汇总
        train_metrics = self.train_metrics.compute()
        avg_losses = {k: v / len(self.train_loader) for k, v in epoch_losses.items()}

        elapsed = time.time() - start_time
        pos_acc = (self._train_pos_correct / self._train_pos_total * 100
                   if self._train_pos_total > 0 else 0.0)
        return {
            "epoch": self.current_epoch,
            "losses": avg_losses,
            "metrics": train_metrics,
            "time": elapsed,
            "position_acc": pos_acc,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """执行验证。"""
        self.model.eval()
        self.val_metrics.reset()
        val_losses = defaultdict(float)
        val_pos_correct = 0
        val_pos_total = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            disease_labels = batch["disease_label"].to(self.device)
            position_labels = batch["position_label"].to(self.device)

            targets = None
            has_det_targets = False
            if self.config.get("detection_enabled", False):
                targets, has_det_targets = self._prepare_detection_targets(batch)

            output = self.model(images, targets=targets)

            # 对没有 bbox 标注的 batch,跳过检测损失
            det_losses = output.get("detection", {}).get("losses")
            if not has_det_targets:
                det_losses = None

            total_loss, loss_dict = self.criterion(
                disease_logits=output["disease_logits"],
                global_logits=output["global_logits"],
                disease_targets=disease_labels,
                detection_losses=det_losses,
                position_logits=output.get("position_logits"),
                position_targets=position_labels,
            )

            for k, v in loss_dict.items():
                val_losses[k] += v

            self.val_metrics.update(
                output["disease_logits"].cpu().numpy(),
                disease_labels.cpu().numpy(),
            )

            # 跟踪位置准确率
            if "position_logits" in output:
                pos_logits = output["position_logits"].cpu()
                pos_targets = position_labels.cpu()
                valid_mask = pos_targets >= 0
                if valid_mask.sum() > 0:
                    pos_pred = pos_logits[valid_mask].argmax(dim=-1)
                    pos_correct = (pos_pred == pos_targets[valid_mask]).sum().item()
                    val_pos_correct += pos_correct
                    val_pos_total += valid_mask.sum().item()

        val_metrics = self.val_metrics.compute()
        avg_losses = {k: v / len(self.val_loader) for k, v in val_losses.items()}
        pos_acc = (val_pos_correct / val_pos_total * 100
                   if val_pos_total > 0 else 0.0)

        # TensorBoard 日志
        if self.writer:
            for k, v in avg_losses.items():
                self.writer.add_scalar(f"val/{k}", v, self.current_epoch)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"val/{k}", v, self.current_epoch)

        return {
            "epoch": self.current_epoch,
            "losses": avg_losses,
            "metrics": val_metrics,
            "position_acc": pos_acc,
        }

    @torch.no_grad()
    def _compute_classification_metrics(self, loader: DataLoader) -> dict:
        """在无采样器的 loader 上计算“真实分布”的疾病分类指标。

        训练 loader 使用了 WeightedRandomSampler，其混淆矩阵/Acc/F1 算在
        重采样后的均衡分布上，与验证集不可比。这里用独立的 train_eval_loader
        （shuffle=False、无 sampler）重算真实训练集指标。
        """
        tracker = MetricsTracker(
            num_classes=len(DISEASE_CLASSES),
            class_names=DISEASE_CLASSES,
        )
        self.model.eval()
        for batch in loader:
            images = batch["image"].to(self.device)
            disease_labels = batch["disease_label"].to(self.device)
            output = self.model(images, targets=None)
            tracker.update(
                output["disease_logits"].cpu().numpy(),
                disease_labels.cpu().numpy(),
            )
        return tracker.compute()

    @torch.no_grad()
    def _compute_val_detection_map(self) -> float:
        """在验证集上计算检测分支的 mAP@0.5。

        需要 val_loader 的 collate 返回 bboxes / lesion_labels;否则返回 0.0。
        供早停/选优联合疾病 F1 使用,让检测分支的退化也能触发干预。
        """
        from training.det_metrics import compute_map
        from config import NUM_LESION_CLASSES

        self.model.eval()
        predictions, ground_truths = [], []
        for batch in self.val_loader:
            if "bboxes" not in batch or "lesion_labels" not in batch:
                return 0.0
            images = batch["image"].to(self.device)
            output = self.model(images, targets=None)
            det = output.get("detection", {})
            boxes = det.get("boxes", [])
            labels = det.get("labels", [])
            scores = det.get("scores", [])

            for i in range(images.shape[0]):
                b = boxes[i].cpu()
                s = scores[i].cpu()
                l = labels[i].cpu()
                if b.numel() > 0:
                    keep = s >= 0.05
                    b, s, l = b[keep], s[keep], l[keep]
                    idx = torch.argsort(s, descending=True)
                    b, s, l = b[idx], s[idx], l[idx]
                predictions.append({
                    "boxes": b.numpy(), "scores": s.numpy(), "labels": l.numpy(),
                })
                gt_b = batch["bboxes"][i].cpu()
                gt_l = batch["lesion_labels"][i].cpu()
                ground_truths.append({
                    "boxes": gt_b.numpy(), "labels": gt_l.numpy(),
                })

        results = compute_map(predictions, ground_truths, NUM_LESION_CLASSES,
                              iou_thresholds=(0.5,))
        return float(results["mAP@0.5"])

    def _log_epoch(self, train_result: dict, val_result: dict, epoch_time: float):
        """将详细的逐轮指标写入 stdout 和日志文件。"""
        train_m = train_result["metrics"]
        val_m = val_result["metrics"]
        epoch = self.current_epoch

        lines = []
        lines.append("=" * 70)
        lines.append(f"Epoch {epoch:3d} | Time: {epoch_time:.1f}s")
        lines.append("-" * 70)
        # 汇总指标
        train_acc = train_m.get("accuracy", 0)
        train_f1 = train_m.get("macro_f1", 0)
        val_acc = val_m.get("accuracy", 0)
        val_f1 = val_m.get("macro_f1", 0)
        val_auc = val_m.get("auc_roc_ovr", float("nan"))
        auc_str = f"{val_auc:.4f}" if not math.isnan(val_auc) else "N/A"
        lines.append(f"  Train | Acc: {train_acc:.4f} | Macro F1: {train_f1:.4f}")
        lines.append(f"  Val   | Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f} | AUC: {auc_str}")
        # 位置准确率
        train_pos = train_result.get("position_acc", 0)
        val_pos = val_result.get("position_acc", 0)
        lines.append(f"  Pos   | Train Acc: {train_pos:.1f}% | Val Acc: {val_pos:.1f}%")
        # 各损失分量(含检测损失,便于观察检测分支是否在学)
        train_losses = train_result.get("losses", {})
        val_losses = val_result.get("losses", {})
        loss_keys = [k for k in ("loss_disease", "loss_global", "loss_detection",
                                 "loss_position", "loss_total") if k in train_losses]
        if loss_keys:
            train_loss_str = "  ".join(
                f"{k}={train_losses.get(k, 0):.4f}" for k in loss_keys)
            val_loss_str = "  ".join(
                f"{k}={val_losses.get(k, 0):.4f}" for k in loss_keys)
            lines.append(f"  Train Loss | {train_loss_str}")
            lines.append(f"  Val   Loss | {val_loss_str}")
        lines.append("-" * 70)
        # 每类 F1
        lines.append("  Per-class F1:")
        header = f"  {'':25s}"
        for cls_name in DISEASE_CLASSES:
            header += f" {cls_name:>22s}"
        lines.append(header)
        train_row = f"  {'Train':25s}"
        val_row = f"  {'Val':25s}"
        for cls_name in DISEASE_CLASSES:
            train_f1_c = train_m.get(f"f1_{cls_name}", 0)
            val_f1_c = val_m.get(f"f1_{cls_name}", 0)
            train_row += f" {train_f1_c:22.4f}"
            val_row += f" {val_f1_c:22.4f}"
        lines.append(train_row)
        lines.append(val_row)
        lines.append("-" * 70)
        # 训练集混淆矩阵
        train_cm = train_m.get("confusion_matrix", [])
        if train_cm:
            lines.append("  Train Confusion Matrix:")
            lines.append(self._format_cm(train_cm))
        # 验证集混淆矩阵
        val_cm = val_m.get("confusion_matrix", [])
        if val_cm:
            lines.append("  Val Confusion Matrix:")
            lines.append(self._format_cm(val_cm))
        lines.append("=" * 70)
        lines.append("")

        block = "\n".join(lines)
        print(block)
        if self.log_file:
            self.log_file.write(block)
            self.log_file.flush()

    def _format_cm(self, cm: list) -> str:
        """将混淆矩阵格式化为对齐的字符串表格。"""
        n = len(cm)
        lines = []
        # 列标题
        header = " " * 12 + "".join(f"{cls:>8s}" for cls in DISEASE_CLASSES[:n])
        lines.append(header)
        for i, row in enumerate(cm):
            row_str = "".join(f"{v:8d}" for v in row)
            row_sum = sum(row)
            lines.append(f"    {DISEASE_CLASSES[i]:8s}  {row_str}  | {row_sum}")
        return "\n".join(lines)

    def _prepare_detection_targets(self, batch: dict) -> tuple[list[dict], bool]:
        """从 batch 准备检测目标。

        返回 (targets, has_boxes) —— 始终返回列表(不会是 None)。
        has_boxes 为 True 表示 batch 中至少一张图像有 bbox 标注。
        """
        targets = []
        has_boxes = False
        for i in range(len(batch["image"])):
            bboxes = batch["bboxes"][i]
            labels = batch["lesion_labels"][i]

            target = {
                "boxes": bboxes.to(self.device),
                "labels": labels.to(self.device),
            }
            targets.append(target)
            if len(bboxes) > 0:
                has_boxes = True
        return targets, has_boxes

    def fit(self) -> dict:
        """完整训练循环,带早停。"""
        print(f"\n{'='*60}")
        print(f"Training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Max epochs: {self.config['max_epochs']}")
        print(f"Early stopping patience: {self.config['early_stopping_patience']}")
        print(f"{'='*60}\n")

        # 打开详细日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = LOG_DIR / f"training_detail_{timestamp}.txt"
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        self.log_file.write(f"Training Log — {timestamp}\n")
        self.log_file.write(f"Device: {self.device}\n")
        self.log_file.write(f"Train samples: {len(self.train_loader.dataset)}\n")
        self.log_file.write(f"Val samples: {len(self.val_loader.dataset)}\n")
        self.log_file.write(f"Config: {self.config}\n")
        self.log_file.write("=" * 70 + "\n\n")
        self.log_file.flush()

        history = {"train": [], "val": []}

        try:
            for epoch in range(self.start_epoch, self.config["max_epochs"]):
                self.current_epoch = epoch
                epoch_start = time.time()

                train_result = self.train_epoch()
                # 用真实分布（无采样器）重算训练集分类指标，替换采样器口径
                if self.train_eval_loader is not None:
                    train_result["metrics"] = self._compute_classification_metrics(
                        self.train_eval_loader
                    )
                val_result = self.validate()
                epoch_time = time.time() - epoch_start

                history["train"].append(train_result)
                history["val"].append(val_result)

                # 详细日志输出
                self._log_epoch(train_result, val_result, epoch_time)

                # 打印本轮摘要
                train_f1 = train_result["metrics"].get("macro_f1", 0)
                val_f1 = val_result["metrics"].get("macro_f1", 0)
                val_acc = val_result["metrics"].get("accuracy", 0)

                # 检测分支 mAP(仅当启用检测时计算;用于联合选优)
                det_map = 0.0
                if self.config.get("detection_enabled", False):
                    det_map = self._compute_val_detection_map()

                # 联合分数:疾病 F1 + 检测 mAP,兼顾分类与检测。
                combined_score = val_f1 + det_map

                print(f"Epoch {epoch:3d} | "
                      f"Train F1: {train_f1:.4f} | "
                      f"Val F1: {val_f1:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Det mAP: {det_map:.4f} | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

                # 早停判断:以联合分数选优,检测退化也会触发干预
                if combined_score > self.best_combined_score:
                    self.best_combined_score = combined_score
                    self.best_val_f1 = val_f1
                    self.best_det_map = det_map
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0
                    self.save_checkpoint("best.pth")
                else:
                    self.epochs_no_improve += 1

                if self.epochs_no_improve >= self.config["early_stopping_patience"]:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {self.epochs_no_improve} epochs)")
                    break
        finally:
            # 确保日志文件关闭
            if self.log_file:
                self.log_file.write(
                    f"\nTraining complete. Best val F1: {self.best_val_f1:.4f}, "
                    f"Best det mAP: {self.best_det_map:.4f} "
                    f"at epoch {self.best_epoch}\n")
                self.log_file.close()
                print(f"\nDetailed log saved to: {self.log_file_path}")

        # 保存最终模型
        self.save_checkpoint("last.pth")

        print(f"\nTraining complete. Best val F1: {self.best_val_f1:.4f}, "
              f"Best det mAP: {self.best_det_map:.4f} at epoch {self.best_epoch}")

        if self.writer:
            self.writer.close()

        return history

    def save_checkpoint(self, filename: str):
        """保存模型 checkpoint。

        仅保存模型权重（不含 optimizer/scheduler 状态）。AdamW 的优化器状态
        约为模型参数的 2 倍，会显著增大内存/磁盘占用（本机仅 8GB 内存，
        torch.save 的序列化缓冲曾导致训练进程被系统 OOM 杀掉）。推理只需要
        model_state_dict，因此这里精简保存。
        """
        path = CHECKPOINT_DIR / filename
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_f1": self.best_val_f1,
            "best_det_map": self.best_det_map,
            "config": self.config,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, filename: str) -> int:
        """加载模型 checkpoint。返回 epoch 序号。"""
        path = CHECKPOINT_DIR / filename
        if not path.exists():
            print(f"  Checkpoint not found: {path}")
            return 0

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # 旧版 checkpoint 可能包含优化器/调度器状态；新版为节省内存不保存，
        # 缺失时跳过（仅恢复模型权重，优化器从当前学习率重新开始）。
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_f1 = checkpoint.get("best_val_f1", 0)
        self.best_det_map = checkpoint.get("best_det_map", 0.0)
        self.best_combined_score = self.best_val_f1 + self.best_det_map
        self.current_epoch = checkpoint.get("epoch", 0)
        # 续训从 checkpoint 的下一个 epoch 开始（仅加载权重，优化器/调度器重置）
        self.start_epoch = self.current_epoch + 1

        print(f"  Loaded checkpoint: {path} (epoch {self.current_epoch}); "
              f"continuing from epoch {self.start_epoch}")
        return self.current_epoch
