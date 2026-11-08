import json
import os

import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from src.dataset import BTXRDDataset
from src.losses import MultiTaskLoss
from src.model import BoneMTL
from src.trainer import train
from src.transforms import build_transforms
from src.utils import get_device, load_checkpoint, load_config, set_seed


def build_loaders(cfg: dict, train_tf, val_tf):
    data_dir = cfg['data']['data_dir']
    img_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_split.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_split.csv'))

    loader_kw = dict(
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
    )
    train_loader = DataLoader(
        BTXRDDataset(train_df, img_dir, mask_dir, train_tf),
        shuffle=True, **loader_kw,
    )
    val_loader = DataLoader(
        BTXRDDataset(val_df, img_dir, mask_dir, val_tf),
        shuffle=False, **loader_kw,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    return train_loader, val_loader


def resume(model, optimizer, save_dir: str) -> int:
    for name in ('last.pth', 'best.pth'):
        path = os.path.join(save_dir, name)
        if os.path.exists(path):
            epoch, metrics = load_checkpoint(path, model, optimizer)
            print(f"Resumed from {name} epoch {epoch} "
                  f"(val_dice: {metrics.get('dice', 'N/A')})")
            return epoch
    return 0


def main():
    cfg = load_config('configs/default.yaml')
    device = get_device()
    set_seed(cfg['data']['seed'])

    train_tf, val_tf = build_transforms(cfg)
    train_loader, val_loader = build_loaders(cfg, train_tf, val_tf)

    model = BoneMTL(
        num_tumor_types=cfg['model']['num_classes'],
        pretrained=cfg['model']['pretrained'],
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total | {trainable:,} trainable")

    tc = cfg['training']
    criterion = MultiTaskLoss(
        tier3_weights=tc['tier3_weights'],
        tier2_pos_weight=tc['tier2_pos_weight'],
        lambda_tier1=tc['lambda_tier1'],
        lambda_tier2=tc['lambda_tier2'],
        lambda_tier3=tc['lambda_tier3'],
        lambda_seg=tc['lambda_seg'],
        focal_gamma=tc['focal_gamma'],
        lambda_aux=tc['lambda_aux'],
        label_smoothing=tc.get('label_smoothing', 0.1),
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tc['learning_rate'],
        weight_decay=tc['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5,
    )

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=tc['swa_lr'], anneal_epochs=5)

    start_epoch = resume(model, optimizer, cfg['checkpoint']['save_dir'])

    history = train(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        swa_model, swa_scheduler, tc['swa_start'],
        device, cfg,
        start_epoch=start_epoch,
    )

    os.makedirs('results', exist_ok=True)
    with open('results/history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()