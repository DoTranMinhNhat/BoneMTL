# prepare_data.py
import os
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.dataset import BTXRDDataset
from src.utils   import load_config, set_seed

cfg      = load_config('configs/default.yaml')
set_seed(cfg['data']['seed'])

DATA_DIR = cfg['data']['data_dir']
IMG_DIR  = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
XLSX     = os.path.join(DATA_DIR, 'dataset.xlsx')

df  = pd.read_excel(XLSX)
idx = np.random.permutation(len(df))
n   = len(df)

train_df = df.iloc[idx[:int(n*0.70)]].reset_index(drop=True)
val_df   = df.iloc[idx[int(n*0.70):int(n*0.85)]].reset_index(drop=True)
test_df  = df.iloc[idx[int(n*0.85):]].reset_index(drop=True)

train_df.to_csv(os.path.join(DATA_DIR, 'train_split.csv'), index=False)
val_df.to_csv(  os.path.join(DATA_DIR, 'val_split.csv'),   index=False)
test_df.to_csv( os.path.join(DATA_DIR, 'test_split.csv'),  index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

val_tf = A.Compose([
    A.Resize(cfg['data']['img_size'], cfg['data']['img_size']),
    A.Normalize(mean=cfg['data']['mean'], std=cfg['data']['std']),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

ds     = BTXRDDataset(train_df, IMG_DIR, MASK_DIR, val_tf)
sample = ds[0]
print(f"image: {sample['image'].shape} | mask: {sample['mask'].shape} | tier1: {sample['tier1']}")