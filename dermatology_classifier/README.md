# dermatology_classifier

纯**皮肤疾病三分类**模型（湿疹 / 类天疱疹 / 荨麻疹），不包含皮损检测与标注。

疾病标签完全由图片所在目录名派生，**不读取任何 `.json` 标注文件**。
训练数据来自 `../labeled/` 与 `../unlabeled/` 两个目录（按文件名 stem 去重后合并）。

## 目录结构

```
labeled/                  # 含 .json 标注的图片（本项目忽略标注，只用图片+目录标签）
    湿疹/ 类天疱疹/ 荨麻疹/
unlabeled/                # 仅图片，无标注
    湿疹/ 类天疱疹/ 荨麻疹/
dermatology_classifier/
    config.py             # 路径、类别、默认超参数
    data.py               # 数据扫描 + 去重合并 + 患者分层划分 + Dataset
    transforms.py         # 训练/验证增强
    model.py              # torchvision backbone + MLP 分类头
    train.py              # 训练入口
    evaluate.py           # 测试集评估
    predict.py            # 单图推理
```

## 安装

```bash
pip install -r requirements.txt
```

首次训练会自动联网下载 ImageNet 预训练权重；离线环境下可先在有网机器缓存，
再传 `--offline`。

## 训练

```bash
python train.py                          # 默认 convnext_tiny, 224px, 30 epochs
python train.py --image_size 384 --epochs 50
python train.py --backbone resnet50
python train.py --augmentation standard
python train.py --offline                # 只用缓存权重
```

数据会按 `(疾病, 患者)` 做患者级分层划分（默认 train/val/test = 70/15/15）。
产物：

- `outputs/checkpoints/best.pth` —— 最佳模型（按 val macro-F1 选优）
- `outputs/history.json` —— 每轮指标
- `outputs/splits.pkl` —— 划分缓存（评估时复现同一测试集）

## 评估 / 推理

```bash
python evaluate.py                                   # 在 held-out 测试集上评估
python predict.py --image path/to/img.jpg            # 单图推理
```

## 说明

- 三分类存在类别不平衡（湿疹样本最多），训练用逆频率类别权重 + 交叉熵；
  backbone 与分类头使用差异化学习率（头 10×）。
- 训练/验证只读图片与目录标签，验证/推理无需任何 `.json` 文件。
- 图片按 stem 去重，处理了 labeled 与 unlabeled 之间的重叠；带 `_2` 等后缀的
  近似重复图片未做内容级去重，如需可后续加 MD5 内容哈希。
