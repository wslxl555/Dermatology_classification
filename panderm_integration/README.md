# panderm_integration

**PanDerm 权重就绪后**，把 PanDerm ViT 特征接入本项目两条路径：

1. **linear probe**（`linear_probe.py`）—— 冻结 PanDerm ViT，抽 CLS 特征 → 线性分类器。
2. **检测骨干接入**（`detector_backbone.py`）—— 把 ViT patch 特征当特征图，接
   torchvision Faster R-CNN / FCOS 检测头（现有检测工程同款头）。

纯脚本，当前环境无需权重/timm 也能用 `--dummy` 验证流水线。

## 前置条件

| 依赖 | 说明 |
|---|---|
| PanDerm 权重 | 尚未下载（见下） |
| `timm` | 加载 PanDerm 的 `modeling_finetune.py` 需要（当前未安装） |
| `torch`/`torchvision`/`sklearn`/`cv2`/`PIL` | 已具备 |

## 下载 PanDerm 权重（网络就绪后）

| 模型 | 文件名 | 地址 |
|---|---|---|
| PanDerm_Base (ViT-B/16) | `panderm_bb_data6_checkpoint-499.pth` | [Google Drive](https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view) |
| PanDerm (ViT-L/16) | `panderm_ll_data6_checkpoint-499.pth` | [Google Drive](https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view) |

放到 `PanDerm/pretrain_weight/`，脚本会自动搜索；或用 `--checkpoint` 显式指定。
若没有 GPU，优先用 `--variant base`（ViT-B/16，CPU 可勉强跑推理）。

```bash
pip install timm
```

## 1. Linear probe

```bash
# 权重就绪后：整图 → 疾病 4 分类
python -m panderm_integration.linear_probe \
    --checkpoint PanDerm/pretrain_weight/panderm_bb_data6_checkpoint-499.pth \
    --variant base --task disease --classes 4 --include_unlabeled

# 病灶 crop → 皮损类型（复用 dermatology_classifier 的 lesion_crops）
python -m panderm_integration.linear_probe \
    --checkpoint PanDerm/pretrain_weight/panderm_bb_data6_checkpoint-499.pth \
    --variant base --task lesion_type

# 无权重：验证整条流水线可跑（假编码器，无需 timm/权重）
python -m panderm_integration.linear_probe --dummy --task disease --classes 4
python -m panderm_integration.linear_probe --dummy --task lesion_type
```

- 特征按 `panderm_integration/cache/*.npz` 缓存，二次运行直接读缓存。
- 用**患者级** Stratified K-fold 评估，避免同一患者跨 train/test 泄漏。
- 指标：Accuracy / Macro-F1 / AUC（LogisticRegression，`class_weight='balanced'`）。

## 2. 检测骨干接入

```bash
# 权重就绪后：构建 FasterRCNN / FCOS（PanDerm ViT 骨干），跑一次前向验证形状
python -m panderm_integration.detector_backbone \
    --checkpoint PanDerm/pretrain_weight/panderm_bb_data6_checkpoint-499.pth \
    --variant base --head fasterrcnn --num_classes 16 --check
```

`num_classes=16` = 15 前景皮损类 + 1 背景（与 `dermatology_detection/config.py`
的 15 类一致）。

## 重要限制（务必读）

1. **原版 ViT 固定 224×224、特征图仅 14×14**，对小病灶太粗。`detector_backbone.py`
   是「可跑的最小正确范式」，用于验证接法；要精度应改用 mmdet ViT-FPN
   （Derm1M 代码已带 `mask_rcnn_vit_fpn` / `cascade_mask_rcnn_vit_fpn` 配置），或
   插值 pos_embed 升分辨率 / 用窗口注意力。
2. **linear probe 与检测都未在本环境跑过真实权重**（网络受限 + 无 timm），
   脚本按 PanDerm 官方 `linear_eval.py` 的约定写（CLS 特征 + LogisticRegression、
   Resize256→CenterCrop224、ImageNet 归一化）。
3. 病种标签来自目录名（`labeled/湿疹` 等），与 `derm1m_diagnosis` 的 4 病种一致；
   `--classes 3` 可退化为不含接触性皮炎的三分类。
