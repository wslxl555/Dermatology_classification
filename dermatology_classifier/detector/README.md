# 病灶检测器（可解释性）

在整图疾病分类器（`dermatology_classifier`）之外，训一个**病灶检测器**，为分类结果提供
可解释性：一张图不仅输出"什么病"，还输出"检出哪些病灶、几个、长什么样"。

## 为什么单独做检测，而不是多任务

之前晚融合实验已证明：形态特征对**微调后的分类器是冗余的**（融合不涨 F1）。但检测的价值
不在提分类准确率，而在**可解释性**——把"检出大疱"作为诊断依据展示出来。所以这里检测器与
分类器解耦，各用各的 backbone，互不影响。

## 检测头（可自由切换）

| head | 类型 | 说明 |
|---|---|---|
| `fcos` | 检测（torchvision） | anchor-free，小目标友好（默认） |
| `retinanet` | 检测（torchvision） | anchor-based，类 FCOS |
| `faster_rcnn` | 检测（torchvision） | 两阶段，精度最高、最慢 |

**分割式检测头**（`attention_unet` / `mamba` / `sam_adapter` / `transunet` / `transcascade`）
已实现在 `dermatology_detection` 工程。由于两个工程有顶层 `config.py`/`models/` 模块名冲突，
不能直接 import，改用子进程桥接：

```bash
python detector/train_segmentation.py --head attention_unet --epochs 20 --offline
```

## 类别（10 类）

诊断关键类全部保留（尤其 `bulla` 大疱，是类天疱疹的核心信号），低频无诊断价值的并入 `other`：

```
macule patch papule plaque vesicle scale crust erosion bulla other
```

## 病灶特征描述（大小/形状/颜色/纹理）

`lesion_features.py` 从检出框内像素提取每个病灶的：
- **大小**：宽×高、占图比例
- **形状**：长宽比 → 圆形/椭圆形/长条/不规则
- **颜色**：红度/亮度 → 红色/棕褐/肤色/深褐
- **纹理**：灰度方差、Laplacian 梯度能量 → 表面粗糙度、边界清晰度

阈值是启发式的（0-255 像素值），后续可用真实物理尺度（dermoscopy 标尺）替换。

## 用法

```bash
# 1) 训练检测器（可切换 head）
python detector/train_detector.py --head fcos --epochs 20 --offline
python detector/train_detector.py --head faster_rcnn --image_size 1024

# 2) 可解释性推理（分类 + 检测 + 病灶特征描述）
python detector/predict_interpretable.py --image /path/to/img.jpg
python detector/predict_interpretable.py --dir /path/to/folder
```

## 依赖

- 检测头用 torchvision（`fasterrcnn_resnet50_fpn_v2` 等），backbone 为 ResNet50。
- 离线训练前需先缓存 ResNet50 权重：
  `python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"`
