"""
纯分类数据增强（无 bbox_params，不涉及 .json 标注）。
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, level: str = "medical") -> A.Compose:
    """训练增强。

    level='medical'：保守增强（适合医学图像，不扭曲皮损形态）。
    level='standard'：更激进增强。
    level='none'：仅 resize + normalize。
    """
    h = w = image_size

    if level == "none":
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    if level == "medical":
        return A.Compose([
            A.Resize(h, w),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.95, 1.05),
                     rotate=(-15, 15), p=0.5),
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(int(h * 0.02), int(h * 0.1)),
                            hole_width_range=(int(w * 0.02), int(w * 0.1)),
                            fill=0, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    if level == "standard":
        return A.Compose([
            A.Resize(h, w),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1),
                     rotate=(-30, 30), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    raise ValueError(f"未知增强级别: {level}")


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """验证/测试变换（无增强）。"""
    h = w = image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
