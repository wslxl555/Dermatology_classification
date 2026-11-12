# 病灶检测器（可解释性）

在整图疾病分类器（`dermatology_classifier`）之外，训一个**病灶检测器**，为分类结果提供
可解释性：一张图不仅输出"什么病"，还输出"检出哪些病灶、几个"（例如"预测类天疱疹，因检出 3 个大疱"）。

## 为什么单独做检测，而不是多任务

之前晚融合实验已证明：形态特征对**微调后的分类器是冗余的**（融合不涨 F1）。但检测的价值
不在提分类准确率，而在**可解释性**——把"检出大疱"作为诊断依据展示出来。所以这里检测器与
分类器解耦，各用各的 backbone，互不影响。

## 检测头（可自由切换）

用 torchvision 内置检测模型，接口统一，`--head` 一键切换：

| head | 说明 |
|---|---|
| `fcos` | anchor-free，小目标友好（默认） |
| `retinanet` | anchor-based，类 FCOS |
| `faster_rcnn` | 两阶段，精度通常最高、最慢 |

## 类别（10 类）

诊断关键类全部保留（尤其 `bulla` 大疱，是类天疱疹的核心信号），低频无诊断价值的并入 `other`：

```
macule patch papule plaque vesicle scale crust erosion bulla other
```

## 用法

```bash
# 1) 训练检测器（可切换 head）
python detector/train_detector.py --head fcos --epochs 20 --offline
python detector/train_detector.py --head faster_rcnn --image_size 1024

# 2) 可解释性推理（分类 + 检测联合）
python detector/predict_interpretable.py --image /path/to/img.jpg
python detector/predict_interpretable.py --dir /path/to/folder
```

## 依赖

- 检测头用 torchvision（`fasterrcnn_resnet50_fpn_v2` 等），backbone 为 ResNet50。
- 离线训练前需先缓存 ResNet50 权重：
  `python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"`
