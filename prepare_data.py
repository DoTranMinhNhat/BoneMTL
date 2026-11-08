import os

import pandas as pd

from src.dataset import BTXRDDataset
from src.splits import group_split
from src.transforms import build_transforms
from src.utils import load_config


def main():
    cfg = load_config('configs/default.yaml')

    data_dir = cfg['data']['data_dir']
    table = pd.read_excel(os.path.join(data_dir, 'dataset.xlsx'))

    splits = group_split(table, seed=cfg['data']['seed'])
    for name, df in splits.items():
        df.to_csv(os.path.join(data_dir, f'{name}_split.csv'), index=False)

    print(" | ".join(f"{name}: {len(df)}" for name, df in splits.items()))

    groups = {name: set(df['group']) for name, df in splits.items()}
    shared = (
        (groups['train'] & groups['val'])
        | (groups['train'] & groups['test'])
        | (groups['val'] & groups['test'])
    )
    print(f"Lesion groups shared between splits: {len(shared)}")

    _, val_tf = build_transforms(cfg)
    ds = BTXRDDataset(
        splits['train'],
        os.path.join(data_dir, 'images'),
        os.path.join(data_dir, 'masks'),
        val_tf,
    )
    sample = ds[0]
    print(f"image: {sample['image'].shape} | mask: {sample['mask'].shape} | "
          f"tier1: {sample['tier1']}")


if __name__ == '__main__':
    main()