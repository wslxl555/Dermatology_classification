# 病灶 crop 分类（两段式）

在已有整图分类器之外，利用 `labeled/` 里 **344 个 LabelMe 病灶标注 JSON**
（丘疹/斑块/水疱/大疱/鳞屑/糜烂… 的锚框 + 形态标签），把「病灶形态」纳入疾病判断。

## 思路

三类病的鉴别信号本质在病灶形态：

| 病 | 关键病灶 |
|---|---|
| 类天疱疹 | 大疱 Bulla / 水疱 Vesicle |
| 荨麻疹 | 风团/丘疹 Papule/Plaque，通常无鳞屑 |
| 湿疹 | 丘疹 + 水疱 + 鳞屑 + 苔藓化 |

两段式：
1. **crop 形态分类器**（`train_lesion_classifier.py`）：裁剪病灶 → 判断每个病灶形态。
2. **形态直方图 → 疾病**（`predict_by_lesions.py`）：把一张图内病灶形态的**比例直方图**
   当作特征，训练一个轻量疾病分类器，患者级 5-fold 交叉验证。

## 文件

- `lesion_crops.py`：核心工具——解析 LabelMe JSON、裁剪病灶、形态标签解析。
- `train_lesion_classifier.py`：训练 crop 形态分类器（复用 `model.build_model` / `train.run_epoch`）。
- `predict_by_lesions.py`：形态直方图 → 疾病（逻辑回归 + 患者级 CV）。
- `late_fusion.py`：晚融合对照实验——CNN 特征 / 形态直方图 / 两者拼接，患者级 5-fold 对拍。
  - 默认用冻结 ImageNet backbone（无泄漏）；`--checkpoint best.pth` 改用微调模型（上界）。
- `run_fusion.sh`：一键跑完整融合流程（见下）。

## 一键运行

```bash
cd "病灶 crop 分类"
bash run_fusion.sh
```

流程：整图 CNN 训练（已有 best.pth 则跳过）→ 病灶形态分类器 → 晚融合对照（冻结 + 微调两种）
→ 两段式疾病预测。

## 已得结论（冻结 swin_t + GT 形态，患者级 5-fold CV）

| 模型 | Accuracy | Macro-F1 | AUC |
|---|---|---|---|
| CNN 特征（冻结） | 0.7909 | 0.7999 | 0.9039 |
| 形态直方图（GT） | 0.8045 | 0.8199 | 0.9170 |
| **CNN+形态（晚融合）** | **0.8273** | **0.8366** | **0.9400** |

形态是视觉特征之外的**互补信息**（融合同时超过两者）。用微调 CNN 的融合见 `late_fusion.py --checkpoint`。


## 用法

```bash
# 1) 先做个「上界」实验：直接用 GT 形态标签 → 疾病（无需训练任何模型，几秒出结果）
python predict_by_lesions.py --use-gt

# 2) 训练 crop 形态分类器
python train_lesion_classifier.py --backbone swin_t --image_size 224 --epochs 20 --offline

# 3) 用训练好的 crop 分类器预测形态 → 疾病（真实可部署的两段式）
python predict_by_lesions.py --checkpoint ../outputs/checkpoints/lesion_morph.pth
```

> 脚本放在带空格的子目录里，运行时 `sys.path` 已自动指向上层 `dermatology_classifier`，
> 因此能 import `config` / `model` / `train` 等。直接在子目录内 `python xxx.py` 即可。

## 形态标签解析约定

- LabelMe 的 label 形如 `Papule（丘疹）+Vesicle（水疱）`，用正则提取英文形态词。
- 复合标签（`+`）与不确定标签（`/`）都解析成形态集合，再按**诊断优先级**取单标签：
  `大疱 > 脓疱 > 水疱 > 苔藓化 > 鳞屑 > 糜烂 > 皲裂 > 抓痕 > 痂 > 斑丘疹 > 丘疹 > 斑块 > 斑片 > 斑疹 > 瘢痕`。
- 优先级把「诊断价值高」的形态（大疱/水疱/鳞屑…）排在前面，复合标签会被归到更关键的形态。

## 已知限制

- **标注只覆盖 labeled/**：unlabeled/ 无 JSON。要做全量，需先用 `dermatology_detection`
  的检测模型对 unlabeled 生成伪框。
- **crop 形态分类器是单标签**：复合形态被压成单标签，损失一部分信息；后续可改成多标签 BCE。
- **随机 crop 划分**：形态分类器训练用随机划分（同图 crop 可能跨 train/val），
  它是中间模型，最终以 `predict_by_lesions.py` 的**患者级 CV** 为准。
