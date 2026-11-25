# derm1m_diagnosis

**Derm1M 对齐的「皮损类型 + 数量 → 皮肤疾病鉴别诊断」离线本体推理包。**

在已有皮损检测（`dermatology_detection`）之上，用 Derm1M 的临床本体知识，
把「检出哪些皮损、各几个」转化为**可解释的鉴别诊断排序**，回答"为什么判这个病"。

纯 Python，不依赖 torch / 网络，可直接在 CPU 上跑。

## 定位

整个诊断流水线分三步，本包负责**第三步**：

```
① 检测皮损区域        dermatology_detection/inference/detect_lesions.py
② 认准皮损类型 + 计数  （检测输出自带 label 与数量）
③ Derm1M 决定诊断     ★ 本包 derm1m_diagnosis
```

## 为什么是"用 Derm1M 决定"

Derm1M 的核心资产是一套**与临床本体对齐**的 130 个临床概念（皮损形态、颜色、
皮肤镜特征等）。本包：

- 直接复用 Derm1M 的概念名（`bulla`/`vesicle`/`scale`/`wheal`/`lichenification`…），
  并读取本地 `Derm1M/concept_annotation/term_lists/ConceptTerms.json` 解析同义词
  （`blister`→bulla、`scaly`→scale、`bullae`→bulla…）。
- 依据标准皮肤病学，把「皮损形态组合」映射到病种，例如 Derm1M 的 `wheal` 概念
  直接关联荨麻疹；`bulla` 是类天疱疹的核心信号。

## 目录

```
derm1m_diagnosis/
    ontology.py        # 15 类皮损 + 4 病种的本体（Derm1M 概念对齐）
    diagnose.py        # 推理引擎（计数饱和 + 加权证据 + 鉴别诊断排序）
    cli.py             # 命令行入口
    test_diagnose.py   # 冒烟测试 / 典型病例示例
    README.md
```

## 快速开始

```bash
# 1) 直接给定皮损计数（离线演示 / 快速验证）
python -m derm1m_diagnosis.cli --counts "bulla:3,erosion:1,crust:1"

# 2) 消费检测器输出（detect_lesions.py --out 的 JSON）
python dermatology_detection/inference/detect_lesions.py \
    --checkpoint dermatology_detection/outputs/checkpoints/best.pth \
    --dir images/ --out det.json
python -m derm1m_diagnosis.cli --detection-json det.json
```

Python 调用：

```python
from derm1m_diagnosis import diagnose, format_report

result = diagnose({"bulla": 3, "erosion": 1, "crust": 1})
print(format_report(result))
# 结论: 最符合「类天疱疹」
```

## 皮损类别（与检测器输出严格一致）

15 类，与 `dermatology_detection/config.py` 的 `LESION_CLASSES` 完全一致：

```
macule patch papule plaque nodule vesicle bulla pustule
scale crust erosion excoriation scar lichenification fissure
```

> 兼容 `dermatology_classifier/detector` 的旧 10 类词汇：旧词汇里的 `other`
> （结节/脓疱/抓痕/瘢痕/苔藓化/皲裂 的合并桶）也会被识别，作为低信息证据参与。

## 本体（4 病种）

| 病种 | 关键损害 | 支持损害 | 反对损害 |
|---|---|---|---|
| 类天疱疹 | 大疱(3.0) | 糜烂/痂/水疱/斑疹/斑块 | — |
| 荨麻疹 | 风团≈斑块/丘疹(1.5) | 斑疹(红斑) | 鳞屑/痂/大疱/糜烂/苔藓化/皲裂/抓痕 |
| 湿疹 | —（多形性组合） | 鳞屑/水疱/苔藓化/丘疹/糜烂/痂/抓痕 | 大疱(弱反对) |
| 接触性皮炎 | — | 水疱/糜烂/大疱/痂/鳞屑/苔藓化/皲裂/斑疹 | — |

- `风团(wheal)` 是荨麻疹特征性损害，但本项目检测器无独立 `wheal` 类，
  故映射到最接近的 `斑块(plaque)/丘疹(papule)`。
- 权重为启发式、临床可解释；计数通过 `min(count, cap=3)` 饱和，使
  「3 个大疱」比「1 个大疱」证据更强，但不会线性放大 20 倍。

## 计分与输出

- 支持度 = Σ 支持权重·饱和计数 − Σ 反对权重·饱和计数（正=支持，负=反对）。
- 病种间用 softmax 归一化为「相对置信度」（仅相对排序，非校准概率）。
- 全为负分（未检出足够支持性皮损）时返回「证据不足」。

每个病种输出：支持度、支持证据、反对证据、缺失关键损害、临床说明。

## 已知限制

1. **检测 checkpoint 与当前 15 类 config 不匹配**：仓库内 `dermatology_detection/
   outputs/checkpoints/best.pth`（FCOS）等是用旧类别数（8 个前景类）训练的，
   直接 `load_state_dict` 会 size mismatch（cls_logits 9 vs 15）。需按当前 15 类
   config 重新训练检测器后，`detect_lesions.py` 才能输出 15 类标签。
2. **`wheal` 无独立检测类**：荨麻疹的风团被近似到斑块/丘疹，信号较弱；
   如需更强区分，可给检测器新增 `wheal` 类。
3. **本体覆盖 4 病种**：按用户确认范围（接触性皮炎/湿疹/类天疱疹/荨麻疹）。
   扩展到 Derm1M 390 病种需要 `ontology.json`（当前网络受限无法下载）。
4. **支持度非校准概率**：只用于病种间相对排序，不能当作确诊概率。

## 测试

```bash
python derm1m_diagnosis/test_diagnose.py
```

覆盖：典型类天疱疹（大疱为主）、典型荨麻疹（风团样）、典型/慢性湿疹、
急性接触性皮炎、鳞屑+结痂（排除荨麻疹），以及空输入（证据不足）、
中文/Derm1M 概念/同义词归一化。
