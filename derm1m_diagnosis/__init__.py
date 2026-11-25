"""Derm1M 辅助诊断管线。

三段式：
  Stage1 皮损检测 + 类型识别   —— 复用 dermatology_detection 的多任务模型
  Stage2 皮损类型计数          —— 统计每种皮损的数量
  Stage3 疾病诊断决策          —— 符号规则(ontology 概念层) + DermLIP 视觉零样本，融合

详见 pipeline.py / run.py。
"""
