"""
皮肤皮损图像的数据增强与变换流水线。
提供 "medical"、"standard" 和 "none" 三种增强级别。
已针对 albumentations v2 API 兼容性进行更新。
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(image_size: tuple[int, int] = (512, 512),
                         level: str = "medical") -> A.Compose:
    """
    获取训练数据增强流水线。

    level='medical': 适用于医学图像的保守增强。
                     避免扭曲皮损形态。
    level='standard': 更激进的数据增强。
    level='none': 仅 resize + normalize。
    """
    h, w = image_size

    if level == "tile":
        # 滑窗切块训练:不在整图上 resize,而是在原图上随机裁 h×w 的块。
        # 病灶保持原生分辨率(中位 ~83px),避免缩到 512 后只剩 ~14px、
        # 在 stride4 上只有 ~3 个正样本格点而无法学定位。
        # 注意:只做不改变尺度的几何变换(翻转/90° 旋转),不做 Affine 缩放。
        return A.Compose([
            A.RandomCrop(h, w, p=1.0),

            # 颜色 / 亮度(安全,不改变尺度)
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

            # 几何变换(翻转 / 90° 旋转不改变尺度)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),

            # 遮挡(albumentations v2: num_holes_range, hole_height_range, hole_width_range)
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(int(h * 0.02), int(h * 0.1)),
                            hole_width_range=(int(w * 0.02), int(w * 0.1)),
                            fill=0, p=0.3),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc",
                                    label_fields=["class_labels"],
                                    min_visibility=0.1))

    if level == "none":
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

    elif level == "medical":
        return A.Compose([
            A.Resize(h, w),

            # 颜色 / 亮度(对医学图像安全——不会扭曲形态)
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

            # 几何变换(安全——皮肤皮损对方向不敏感)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.95, 1.05),
                     rotate=(-15, 15), p=0.5),

            # 遮挡(albumentations v2: 使用 num_holes_range, hole_height_range, hole_width_range)
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(int(h * 0.02), int(h * 0.1)),
                            hole_width_range=(int(w * 0.02), int(w * 0.1)),
                            fill=0, p=0.3),

            # 模糊(模拟对焦问题)
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

    elif level == "standard":
        return A.Compose([
            A.Resize(h, w),

            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1),
                     rotate=(-30, 30), p=0.5),

            A.CoarseDropout(num_holes_range=(1, 8),
                            hole_height_range=(int(h * 0.02), int(h * 0.1)),
                            hole_width_range=(int(w * 0.02), int(w * 0.1)),
                            fill=0, p=0.3),

            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

    else:
        raise ValueError(f"Unknown augmentation level: {level}")


def get_val_transforms(image_size: tuple[int, int] = (512, 512),
                       level: str = "standard") -> A.Compose:
    """获取验证/测试数据增强流水线(无增强)。

    level='standard': resize + normalize(默认)。
    level='tile':     原生尺度中心裁 h×w(与切块训练保持同一尺度)。
    """
    h, w = image_size
    if level == "tile":
        return A.Compose([
            A.CenterCrop(h, w, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc",
                                    label_fields=["class_labels"],
                                    min_visibility=0.1))
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))


# ---------------------------------------------------------------------------
# 纯疾病分类用的变换（不含 bbox_params，只处理图像，不涉及 .json 标注）
# ---------------------------------------------------------------------------

def get_cls_train_transforms(
    image_size: tuple[int, int] = (224, 224),
    level: str = "medical",
) -> A.Compose:
    """纯分类训练增强。不传 bbox_params，仅对图像做几何/颜色增强。"""
    h, w = image_size
    norm = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    if level == "none":
        return A.Compose([A.Resize(h, w), *norm, ToTensorV2()])

    elif level == "medical":
        return A.Compose([
            A.Resize(h, w),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            *norm,
            ToTensorV2(),
        ])

    elif level == "standard":
        return A.Compose([
            A.Resize(h, w),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-30, 30), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            *norm,
            ToTensorV2(),
        ])

    else:
        raise ValueError(f"Unknown augmentation level: {level}")


def get_cls_val_transforms(image_size: tuple[int, int] = (224, 224)) -> A.Compose:
    """纯分类验证/测试变换（无增强，仅 resize + normalize）。"""
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
