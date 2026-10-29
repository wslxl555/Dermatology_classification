# 皮肤病皮损自动检测与标注模型

基于 `labeled/` 文件夹中的 LabelMe 标注数据，训练一个多任务模型，对皮肤病图片**自动检测皮损（边界框）并标注皮损类型**，同时给出整图的疾病分类（湿疹 / 类天疱疹 / 荨麻疹）。

## 数据

`labeled/{疾病中文名}/` 下每个 `.JPG` 图片对应一个 `.json`（LabelMe 格式）标注：

| 疾病目录 | 疾病（英文） | 图片数 |
|---|---|---|
| 湿疹 | eczema | 200 |
| 类天疱疹 | bullous_pemphigoid | 55 |
| 荨麻疹 | urticaria | 89 |

- 共 344 张已标注图片，4342 个皮损边界框。
- 原始标签 214 种（中英混用、含拼写错误、复合标签如 `Plaque+Scale`），
  由 `data/label_mapping.py` 归一化为 **15 个标准皮损类别**（多标签）：
  macule 斑疹、patch 斑片、papule 丘疹、plaque 斑块、nodule 结节、vesicle 水疱、
  bulla 大疱、pustule 脓疱、scale 鳞屑、crust 痂、erosion 糜烂、excoriation 抓痕、
  scar 瘢痕、lichenification 苔藓化、fissure 皲裂。

## 模型

- 骨干网络：`convnext_tiny`（ImageNet 预训练，权重已本地缓存）。
- 检测头：FCOS 风格 anchor-free（FPN + 共享卷积头），见 `models/detection_head.py`。
- 多任务：疾病分类（3 类）+ 皮损检测（15 类，检测时取每框主类别）。

## 训练

### 本地 CPU（当前机器，无 GPU、仅 8GB 内存）

```bash
cd dermatology_detection
python train_labeled.py --image_size 256 --epochs 10 --batch_size 2 --offline
```

- **内存受限**：本机仅 8GB 内存，`--image_size 320 --batch_size 4` 会在训练中途被系统
  OOM 杀掉。请使用 `--image_size 256 --batch_size 2`（每个 epoch 约 12–15 分钟）。
- `--image_size` 越大精度越高但越慢；本机为 CPU，256 约 12–15 分钟/epoch。
- `--tiny` 快速自检（3 epoch、无检测分支）。
- checkpoint 仅保存模型权重（不含 optimizer 状态），以减小内存/磁盘占用（约 130MB，
  而非含 optimizer 状态时的 400MB+）。
- 输出：`outputs/checkpoints/best.pth`、`last.pth`，`outputs/training_history_labeled.json`。

### HPC / GPU

在含 GPU 的节点上（权重需先缓存或可联网）：

```bash
python train_labeled.py --image_size 512 --epochs 50 --batch_size 8
```

（`train_labeled.py` 会自动在 `cuda` 可用时使用 GPU；原 `run_training_hpc.sh`
是针对旧 `images/` 全量数据的 `train.py` 入口，与 `labeled/` 数据不同，本数据请用 `train_labeled.py`。）

## 训练结果（本机 CPU，256px，10 epoch）

| 指标 | 值 |
|---|---|
| 疾病分类 Val Acc | 0.857 |
| 疾病分类 Val Macro-F1 | 0.856 |
| 疾病分类 Val AUC | 0.912 |
| 最佳 epoch | 3（按 F1 保存于 `best.pth`） |

- 逐类 F1（val）：eczema 0.878 / bullous_pemphigoid 0.875 / urticaria 0.815。
- 疾病分类已收敛良好，可直接用于自动判读病种。
- 检测分支损失在训练中从 9.7 降至 ~5.8，但 10 个 CPU epoch 仍不足以让
  anchor-free 检测头输出可靠检测框（阈值 0.5 下基本无框；低阈值下输出大量
  ~0.3 的低置信框）。**可靠的皮损检测需 GPU 训练 30–50 epoch**。
- 检测/标注管线（检测框 + LabelMe JSON 输出）已完全打通，仅精度受限于本机
  8GB 内存 + CPU。

## 推理（自动检测与标注）

对单张图片或整个文件夹自动检测并标注，输出可视化图 + LabelMe 兼容 JSON：

```bash
# 单张
python -m inference.predict_labeled --image D:/some/path.jpg --checkpoint outputs/checkpoints/best.pth

# 整个文件夹
python -m inference.predict_labeled --image_dir D:/some/folder/ --conf_threshold 0.3
```

输出到图片所在目录的 `predictions/` 子目录：
- `*_pred.jpg`：在原图上绘制的检测框 + 标签 + 疾病分类。
- `*.json`：LabelMe 格式标注，可直接用 Labelme 打开复核。

## 目录结构

```
dermatology_detection/
├── config.py                 # 路径/类别/超参数
├── train_labeled.py          # labeled 数据训练入口
├── data/
│   ├── prepare_labeled_data.py  # 扫描 labeled/、按患者分层划分
│   ├── label_mapping.py         # 原始标签 → 15 类标准化（多标签）
│   ├── dataset.py               # PyTorch Dataset（读取图片+bbox+标签）
│   └── transforms.py            # 数据增强/预处理
├── models/                   # backbone / 检测头 / 多任务模型
├── training/                 # 训练循环 / 损失 / 指标
├── inference/predict_labeled.py  # 自动检测与标注
└── outputs/                  # 划分 / 日志 / 检查点
```

## 重要修复说明（albumentations 2.x）

`data/dataset.py` 与 `data/transforms.py` 曾按 albumentations **1.x** 的“归一化 bbox（0–1）”
约定编写；本机安装的 albumentations **2.0.8** 已改为 **像素坐标 bbox**（输入与输出均为像素值）。
这会导致所有边界框被错误缩放、检测损失恒为 0。已修复为直接传递/读取像素坐标；
并将原 `LongestMaxSize + PadIfNeeded`（其 `PadIfNeeded` 在 2.0.8 下对 bbox 有 bug）
替换为正方形 `Resize`。推理侧检测框按 `原图尺寸 / 输入尺寸` 比例映射回原图像素坐标。
