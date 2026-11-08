import gc
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from src.dataset import BTXRDDataset
from src.losses import MultiTaskLoss
from src.model import BoneMTL
from src.splits import stratify_labels
from src.trainer import train
from src.transforms import build_transforms
from src.utils import get_device, load_config, set_seed

N_FOLDS = 5
CV_PATH = 'results/cv_results.json'

# Name in the fold result : name in the validation metrics
FOLD_METRICS = {
    'dice': 'dice',
    'iou': 'iou',
    'tier1_f1': 'tier1_f1',
    'tier1_auc': 'tier1_auc',
    'tier2_f1': 'tier2_f1',
    'tier2_auc': 'tier2_auc',
    'tier3_f1': 'tier3_f1_macro',
    'tier3_acc': 'tier3_acc',
}


def load_full_df(data_dir: str) -> pd.DataFrame:
    dfs = []
    for split in ('train_split', 'val_split', 'test_split'):
        path = os.path.join(data_dir, f'{split}.csv')
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def load_progress() -> list:
    if not os.path.exists(CV_PATH):
        return []
    with open(CV_PATH) as f:
        return json.load(f).get('folds', [])


def make_loader(df, img_dir, mask_dir, transform, batch_size, shuffle):
    return DataLoader(
        BTXRDDataset(df, img_dir, mask_dir, transform),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True,
    )


def run_fold(fold_num, train_loader, val_loader, cfg, device) -> dict:
    tc = cfg['training']

    model = BoneMTL(
        num_tumor_types=cfg['model']['num_classes'],
        pretrained=cfg['model']['pretrained'],
    ).to(device)

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

    fold_cfg = {
        **cfg,
        'checkpoint': {
            **cfg['checkpoint'],
            'save_dir': os.path.join(
                cfg['checkpoint']['save_dir'], f"fold_{fold_num}"
            ),
        },
    }

    history = train(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        swa_model, swa_scheduler, tc.get('cv_swa_start', tc['swa_start']),
        device, fold_cfg,
    )

    best_val = max(history['val'], key=lambda x: x.get('dice', 0))
    return {
        'fold': fold_num,
        **{name: best_val.get(key, 0) for name, key in FOLD_METRICS.items()},
    }


def main():
    cfg = load_config('configs/default.yaml')
    device = get_device()
    set_seed(cfg['data']['seed'])

    data_dir = cfg['data']['data_dir']
    img_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    batch_size = cfg['data']['batch_size']

    full_df = load_full_df(data_dir)
    print(f"Total samples: {len(full_df)}")

    os.makedirs('results', exist_ok=True)
    results = load_progress()
    done_folds = {r['fold'] for r in results}
    if done_folds:
        print(f"Resuming, folds done: {sorted(done_folds)}")

    if 'group' not in full_df:
        raise SystemExit("Split files have no 'group' column, run prepare_data.py")

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train_tf, val_tf = build_transforms(cfg)

    splits = skf.split(
        full_df, stratify_labels(full_df), groups=full_df['group']
    )
    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        if fold_num in done_folds:
            print(f"Fold {fold_num} already done, skipping.")
            continue

        print(f"\n{'=' * 50}\nFold {fold_num}/{N_FOLDS}\n{'=' * 50}")

        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        print(f"Train: {len(train_df)} | Val: {len(val_df)}")

        train_loader = make_loader(
            train_df, img_dir, mask_dir, train_tf, batch_size, shuffle=True
        )
        val_loader = make_loader(
            val_df, img_dir, mask_dir, val_tf, batch_size, shuffle=False
        )

        fold_result = run_fold(fold_num, train_loader, val_loader, cfg, device)
        results.append(fold_result)
        print(f"\nFold {fold_num} best: Dice={fold_result['dice']:.4f} | "
              f"Tier3 F1={fold_result['tier3_f1']:.4f}")

        with open(CV_PATH, 'w') as f:
            json.dump({'folds': results}, f, indent=2)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'=' * 50}\n{N_FOLDS}-Fold Cross-Validation Results\n{'=' * 50}")
    summary = {'folds': results}
    for name in FOLD_METRICS:
        vals = [r[name] for r in results]
        mean, std = np.mean(vals), np.std(vals)
        summary[name] = {'mean': float(mean), 'std': float(std)}
        print(f"  {name:<12}: {mean:.4f} ± {std:.4f}")

    with open(CV_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {CV_PATH}")


if __name__ == '__main__':
    main()