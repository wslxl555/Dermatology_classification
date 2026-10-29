"""
皮肤病分类模型的配置文件。
集中管理所有路径、超参数和类别定义。
"""

import os
from pathlib import Path

# =============================================================================
# 路径
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_ROOT = PROJECT_ROOT.parent / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 疾病类别(4 类)
# =============================================================================
DISEASE_CLASSES = [
    "eczema",              # 湿疹
    "bullous_pemphigoid",  # 大疱型类天疱疮
    "urticaria",           # 荨麻疹
]

DISEASE_CLASSES_ZH = {
    "eczema": "湿疹",
    "bullous_pemphigoid": "大疱型类天疱疮",
    "urticaria": "荨麻疹",
}

NUM_DISEASE_CLASSES = len(DISEASE_CLASSES)

# =============================================================================
# 标准化皮损标签(9 类 = 8 个高频类 + "other")
# 与 data/label_mapping.py 的 LESION_CLASSES 保持严格一致。
# =============================================================================
# 原发性皮损(高频)
LESION_CLASSES = [
    "macule",          # 斑疹 — 扁平、不可触及的颜色改变
    "patch",           # 斑片 — 较大的斑疹(>1cm)
    "papule",          # 丘疹 — 小而实、隆起(<0.5cm)
    "plaque",          # 斑块 — 较大的扁平隆起(>0.5cm)
    "vesicle",         # 水疱 — 小的含液疱(<0.5cm)
    # 继发性皮损(高频)
    "scale",           # 鳞屑 — 角质层脱落
    "crust",           # 痂 — 干涸的血清/血液/脓液
    "erosion",         # 糜烂 — 表皮部分缺失
    # 低频类别合并
    "other",           # 其他(结节/大疱/脓疱/抓痕/瘢痕/苔藓化/皲裂)
]

NUM_LESION_CLASSES = len(LESION_CLASSES)  # 9

# =============================================================================
# 检测头可选值（供 argparse / 文档使用）
# =============================================================================
DETECTION_HEAD_NAMES = [
    "fcos", "faster_rcnn", "cascade_rcnn", "yolov8", "rtdetr",
    "transunet", "transcascade", "attention_unet", "mamba", "sam_adapter",
]

LESION_CLASSES_ZH = {
    "macule": "斑疹",
    "patch": "斑片",
    "papule": "丘疹",
    "plaque": "斑块",
    "nodule": "结节",
    "vesicle": "水疱",
    "bulla": "大疱",
    "pustule": "脓疱",
    "scale": "鳞屑",
    "crust": "痂",
    "erosion": "糜烂",
    "excoriation": "抓痕",
    "scar": "瘢痕",
    "lichenification": "苔藓化",
    "fissure": "皲裂",
    "other": "其他",
}

# 重映射:原始标签 → 标准化标签
# 由 label_mapping.py 在运行时构建;以下是关键的参考映射。

# =============================================================================
# 目录到疾病映射
# =============================================================================
# 每种疾病类型位于 DATA_ROOT 下的特定目录
DISEASE_DIR_MAP = {
    # 湿疹数据位于 sz/ 下——!入选/ 为入选患者数据，labelme已标注数据/ 为精细标注数据
    "eczema": DATA_ROOT / "sz",
    # 类天疱疮和荨麻疹数据位于各自独立目录
    "bullous_pemphigoid": DATA_ROOT / "tpc",
    "urticaria": DATA_ROOT / "xmz",
}

# 数据质量层级配置（用于 --data_tier 参数）
# "入选": 仅使用高质量入选数据
# "all": 使用全部数据（含不完整数据和 labelme 标注数据）
DATA_TIER_CONFIG = {
    "入选": {
        "eczema_subdirs": ["!入选"],
        "urticaria_subdirs": ["1.入选 荨麻疹 -完整数据"],
        "include_labelme": False,
    },
    "all": {
        "eczema_subdirs": ["!入选", "labelme已标注数据"],
        "urticaria_subdirs": [
            "1.入选 荨麻疹 -完整数据",
            "2.入选 荨麻疹 不完整数据（分类后L+S）",
            "3.入选 荨麻疹 不完整数据（仅有L）",
        ],
        "include_labelme": True,
    },
}

# =============================================================================
# 模型超参数
# =============================================================================
MODEL_CONFIG = {
    # Backbone(骨干网络)
    "backbone": "convnext_tiny",  # convnext_tiny, convnext_small, resnet50, swin_t
    "pretrained": True,
    "pretrained_weights": "IMAGENET1K_V1",  # 或 ConvNeXt 使用 IMAGENET22K

    # 检测分支
    "detection_enabled": True,
    # 检测头可选值：fcos / faster_rcnn / cascade_rcnn / yolov8 / rtdetr
    "detection_head": "fcos",
    "fpn_channels": 256,
    "rpn_pre_nms_top_n_train": 2000,
    "rpn_pre_nms_top_n_test": 1000,
    "rpn_post_nms_top_n_train": 512,
    "rpn_post_nms_top_n_test": 256,
    "box_detections_per_img": 100,

    # 分类头
    "global_hidden_dim": 512,
    "dropout": 0.3,

    # 多任务损失权重
    # 新的 FCOS 检测损失(cls + reg + ctr)本身已经很好地缩放;
    # 权重 1.0 使其与疾病分类损失保持均衡。
    "loss_disease_weight": 1.0,
    "loss_detection_weight": 1.0,
}

TRAINING_CONFIG = {
    # 通用
    "seed": 42,
    "device": "cuda",  # cuda, cpu

    # 数据
    "image_size": (512, 512),  # H x W(高 x 宽)
    "batch_size": 8,
    "num_workers": 4,

    # 数据增强
    "augmentation": "medical",  # medical, standard, none

    # 优化器
    "optimizer": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "lr_scheduler": "cosine_warmup",
    "warmup_epochs": 3,
    "min_lr": 1e-6,

    # 训练
    "max_epochs": 50,
    "early_stopping_patience": 6,
    "gradient_clip_norm": 1.0,

    # 损失
    "focal_alpha": None,  # 根据类别分布自动计算
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,

    # 交叉验证
    "n_folds": 5,
    "val_split": 0.2,

    # 混合精度
    "use_amp": True,

    # 日志
    "log_interval": 10,  # 记录日志的步数间隔
    "eval_interval": 1,  # 评估的 epoch 间隔
}

# 完整配置 = MODEL_CONFIG + TRAINING_CONFIG 合并
FULL_CONFIG = {**MODEL_CONFIG, **TRAINING_CONFIG}

# 快速测试用的小型配置(用于快速验证)
TINY_CONFIG = {
    **FULL_CONFIG,
    "image_size": (224, 224),
    "batch_size": 4,
    "num_workers": 0,
    "max_epochs": 3,
    "detection_enabled": False,
}

# CPU 本地训练:平衡速度与训练时长的中等配置
# 用于本地无 GPU 环境下生成完整训练日志
MEDIUM_CPU_CONFIG = {
    **FULL_CONFIG,
    "device": "cpu",
    "image_size": (224, 224),
    "batch_size": 4,
    "num_workers": 0,
    "max_epochs": 30,
    "early_stopping_patience": 10,
    "detection_enabled": False,
    "use_amp": False,
    "augmentation": "medical",
}

# =============================================================================
# 实验预设配置（Exp 1–6）
# =============================================================================
EXP1_CONFIG = {
    **FULL_CONFIG,
    "backbone": "convnext_tiny",
    "image_size": (224, 224),
    "batch_size": 4,
    "max_epochs": 30,
    "early_stopping_patience": 10,
    "detection_enabled": False,
    "augmentation": "medical",
    "data_tier": "入选",
    "exp_name": "Exp1_CleanBaseline",
}

EXP2_CONFIG = {
    **FULL_CONFIG,
    "backbone": "resnet50",
    "image_size": (224, 224),
    "batch_size": 4,
    "max_epochs": 30,
    "early_stopping_patience": 10,
    "detection_enabled": False,
    "augmentation": "medical",
    "data_tier": "入选",
    "exp_name": "Exp2_ResNet50",
}

EXP3_CONFIG = {
    **FULL_CONFIG,
    "backbone": "convnext_tiny",
    "image_size": (224, 224),
    "batch_size": 4,
    "max_epochs": 30,
    "early_stopping_patience": 10,
    "detection_enabled": False,
    "augmentation": "medical",
    "data_tier": "all",
    "exp_name": "Exp3_AllData",
}

EXP4_CONFIG = {
    **FULL_CONFIG,
    "backbone": "convnext_tiny",
    "image_size": (512, 512),
    "batch_size": 8,
    "max_epochs": 30,
    "early_stopping_patience": 10,
    "detection_enabled": False,
    "augmentation": "medical",
    "data_tier": "入选",
    "exp_name": "Exp4_HighRes",
}

EXP5_CONFIG = {
    **FULL_CONFIG,
    "backbone": "convnext_tiny",
    "image_size": (512, 512),
    "batch_size": 8,
    "max_epochs": 30,
    "early_stopping_patience": 10,
    "detection_enabled": False,
    "augmentation": "standard",
    "data_tier": "入选",
    "exp_name": "Exp5_AugStudy",
}

EXP6_CONFIG = {
    **FULL_CONFIG,
    "backbone": "convnext_tiny",
    "image_size": (512, 512),
    "batch_size": 8,
    "max_epochs": 30,
    "early_stopping_patience": 10,
    "detection_enabled": True,
    "augmentation": "medical",
    "data_tier": "all",
    "exp_name": "Exp6_MultiTask",
}

EXP_CONFIG_MAP = {
    1: EXP1_CONFIG,
    2: EXP2_CONFIG,
    3: EXP3_CONFIG,
    4: EXP4_CONFIG,
    5: EXP5_CONFIG,
    6: EXP6_CONFIG,
}
