import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(cfg: dict):
    size = cfg['data']['img_size']
    mean, std = cfg['data']['mean'], cfg['data']['std']

    train_tf = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.3,
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    val_tf = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    return train_tf, val_tf