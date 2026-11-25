# derm1m_diagnosis —— Derm1M 辅助诊断管线

把「皮损检测 → 类型识别 → 类型计数 → 疾病诊断」串成一条可解释的诊断管线，
用 Derm1M（DermLIP 视觉-语言模型 + 临床概念 ontology）做最终的疾病决策。

## 管线结构

```
图像
 ├─ Stage1 皮损检测 + 类型   复用 dermatology_detection 的 LesionPredictor（FCOS 多任务模型）
 ├─ Stage2 皮损类型计数       lesion_rules.count_lesions()
 └─ Stage3 疾病诊断决策
      信号A 符号规则          lesion_rules.score_from_counts()  （Derm1M ontology 概念层）
      信号B DermLIP 视觉零样本 dermlip.DermLIPClient.zero_shot_disease()
     → 融合(softmax 后按 alpha 加权) → 最终疾病 + 可解释文字
```

## 用法

```bash
# 单张（纯规则模式，无需下载 DermLIP 权重）
python -m derm1m_diagnosis.run --image D:/path/img.jpg --no_dermlip

# 单张（启用 DermLIP 视觉信号）
python -m derm1m_diagnosis.run --image D:/path/img.jpg

# 文件夹批量 + 输出 JSON
python -m derm1m_diagnosis.run --image_dir D:/path/folder/ --out results.json

# 融合权重：alpha=1 只用规则，alpha=0 只用 DermLIP
python -m derm1m_diagnosis.run --image img.jpg --alpha 1.0
```

## 三个信号的含义

| 信号 | 输入 | 依据 | 强弱 |
|---|---|---|---|
| A 符号规则 | 皮损类型 + 数量 | Derm1M ontology 概念层（bulla→类天疱疹、scale/crust→湿疹、无鳞屑→荨麻疹） | 可解释、无需权重 |
| B DermLIP 零样本 | 整图 + 各病灶 crop | DermLIP 图文对齐（CLIP 风格） | 需权重 + 网络 |
| C 文本检索 | 皮损摘要句 | DermLIP 文本编码器 → 诊断描述语料库 | 近似，需验证 |

## 已知问题（重要）

1. **Stage1 检测头不可靠**：`best.pth` 的 FCOS 检测头在 CPU 训练下已坍缩为
   "全图输出 bulla"（100 个框、均值置信度 0.60）。**检测输入不可信，则 Stage2/3 的
   计数与规则结论也会被带偏**。这是训练问题，非代码问题。修复路径：
   - GPU 重训检测头（README 建议 30–50 epoch），或
   - 用 DermLIP 零样本对病灶 crop 做 concept 分类（`zero_shot_lesion_crop`）替代/校准 FCOS。
2. **皮损类别无 wheal（风团）**：荨麻疹的标志性皮损不在 15 类里，规则层对荨麻疹
   只能靠负证据（无鳞屑/痂/糜烂），判别较弱，需靠 DermLIP 视觉信号兜底。
3. **DermLIP 依赖未装**：需 `pip install safetensors timm`（open_clip 已在 Derm1M/src 内），
   且首次运行会从 HuggingFace 下载权重（约 350MB）。不可用时管线自动退化为纯规则模式。
4. **旧 checkpoint 与 15 类体系不兼容**：config 皮损类别已恢复为 15 类，但现有的
   `best.pth` 是旧 9 类训练的，类别索引会错位（例如旧的 index 5 在新列表里变成了 vesicle）。
   检测头需用 15 类重新训练后才能正确使用新的分类体系。

## 文件

- `lesion_rules.py` —— Stage2 计数 + Stage3 符号规则（纯 Python，无依赖）
- `dermlip.py` —— DermLIP 客户端（懒加载 + 优雅降级）
- `pipeline.py` —— 三段编排 + 融合 + 文字诊断
- `run.py` —— CLI

## 附带修复

- `dermatology_detection/models/multi_task_model.py`：`build_model` 新增
  `num_disease_classes` / `num_lesion_classes` 参数。
- `dermatology_detection/inference/predict_labeled.py`：`LesionPredictor` 从 checkpoint
  的 state_dict 推断实际类别数（旧 checkpoint 9 类 vs config 10 类），避免加载失败。
