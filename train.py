# train.py
import os
import json
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.dataset import BTXRDDataset
from src.model   import BoneMTL
from src.losses  import MultiTaskLoss
from src.trainer import train
from src.utils   import load_config, set_seed, get_device


def build_transforms(cfg: dict):
    """Tạo augmentation pipeline cho train và val."""
    sz, mean, std = cfg['data']['img_size'], cfg['data']['mean'], cfg['data']['std']
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.Resize(sz, sz),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})
    val_tf = A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})
    return train_tf, val_tf


def build_loaders(cfg: dict, train_tf, val_tf):
    """Tạo DataLoader cho train và val."""
    data_dir = cfg['data']['data_dir']
    img_dir  = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_split.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_split.csv'))

    kw = dict(
        batch_size  = cfg['data']['batch_size'],
        num_workers = cfg['data']['num_workers'],
        pin_memory  = True,
    )
    train_loader = DataLoader(
        BTXRDDataset(train_df, img_dir, mask_dir, train_tf),
        shuffle=True, **kw,
    )
    val_loader = DataLoader(
        BTXRDDataset(val_df, img_dir, mask_dir, val_tf),
        shuffle=False, **kw,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    return train_loader, val_loader


def main():
    cfg    = load_config('configs/default.yaml')
    device = get_device()
    set_seed(cfg['data']['seed'])

    train_tf, val_tf         = build_transforms(cfg)
    train_loader, val_loader = build_loaders(cfg, train_tf, val_tf)

    model = BoneMTL(
        num_tumor_types = cfg['model']['num_classes'],
        pretrained      = cfg['model']['pretrained'],
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total | {trainable:,} trainable")

    criterion = MultiTaskLoss(
        tier3_weights    = cfg['training']['tier3_weights'],
        tier2_pos_weight = cfg['training']['tier2_pos_weight'],
        lambda_tier1     = cfg['training']['lambda_tier1'],
        lambda_tier2     = cfg['training']['lambda_tier2'],
        lambda_tier3     = cfg['training']['lambda_tier3'],
        lambda_seg       = cfg['training']['lambda_seg'],
        device           = device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg['training']['learning_rate'],
        weight_decay = cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5,
    )

    history = train(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, device, cfg,
    )

    os.makedirs('results', exist_ok=True)
    with open('results/history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()