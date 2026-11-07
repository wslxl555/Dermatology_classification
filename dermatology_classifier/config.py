"""
皮肤病分类项目配置（自包含，纯分类）。

只做三种皮肤病分类（湿疹 / 类天疱疹 / 荨麻疹），不涉及皮损检测与标注。
疾病标签完全由图片所在目录名派生，不读取任何 .json 标注文件。
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据目录（labeled 含 .json 标注但本项目忽略；unlabeled 仅图片）
LABELED_ROOT = PROJECT_ROOT.parent / "labeled"
UNLABELED_ROOT = PROJECT_ROOT.parent / "unlabeled"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# 疾病类别（3 类，顺序固定）
DISEASE_CLASSES = ["eczema", "bullous_pemphigoid", "urticaria"]
DISEASE_CLASSES_ZH = {
    "eczema": "湿疹",
    "bullous_pemphigoid": "类天疱疹",
    "urticaria": "荨麻疹",
}

# 疾病中文目录名 → 英文（与实际目录名严格一致）
DISEASE_CN_TO_EN = {
    "湿疹": "eczema",
    "类天疱疹": "bullous_pemphigoid",
    "荨麻疹": "urticaria",
}

DISEASE_TO_IDX = {name: i for i, name in enumerate(DISEASE_CLASSES)}
NUM_DISEASE_CLASSES = len(DISEASE_CLASSES)

# 默认超参数
DEFAULT_CONFIG = {
    "backbone": "convnext_tiny",
    "pretrained": True,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 1e-4,      # 分类头学习率（从头训练，较大）
    "backbone_lr": 1e-5,        # 预训练 backbone 微调学习率（较小，防止冲掉特征）
    "weight_decay": 5e-4,       # 加大 L2 正则，抑制小数据过拟合
    "label_smoothing": 0.1,     # 标签平滑，缓解过拟合 + 过自信
    "val_size": 0.15,
    "test_size": 0.15,
    "seed": 42,
    "augmentation": "medical",
    "hidden_dim": 512,
    "dropout": 0.5,             # 加大 dropout
    "early_stopping_patience": 8,
}

for _d in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
