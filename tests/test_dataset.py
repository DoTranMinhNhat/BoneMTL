# tests/test_dataset.py
import os
import sys
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.dataset import BTXRDDataset, TUMOR_COLS
from src.utils   import load_config

cfg      = load_config('configs/default.yaml')
DATA_DIR = cfg['data']['data_dir']
IMG_DIR  = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')

tf = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=cfg['data']['mean'], std=cfg['data']['std']),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

df = pd.read_csv(os.path.join(DATA_DIR, 'train_split.csv'))
ds = BTXRDDataset(df, IMG_DIR, MASK_DIR, tf)


def test_len():
    assert len(ds) == len(df)


def test_sample_shapes():
    s = ds[0]
    assert s['image'].shape == torch.Size([3, 256, 256])
    assert s['mask'].shape  == torch.Size([1, 256, 256])
    assert s['tier1'].shape == torch.Size([1])
    assert s['tier2'].shape == torch.Size([1])
    assert s['tier3'].shape == torch.Size([9])


def test_sample_dtypes():
    s = ds[0]
    assert s['image'].dtype == torch.float32
    assert s['mask'].dtype  == torch.float32
    assert s['tier1'].dtype == torch.float32


def test_tier1_values():
    for i in range(20):
        s = ds[i]
        assert s['tier1'].item() in [0.0, 1.0]


def test_tier2_values():
    for i in range(20):
        s = ds[i]
        assert s['tier2'].item() in [-1.0, 0.0, 1.0]


def test_tier2_minus1_when_no_tumor():
    for i in range(len(ds)):
        s = ds[i]
        if s['tier1'].item() == 0.0:
            assert s['tier2'].item() == -1.0
            break


def test_mask_values():
    for i in range(10):
        s = ds[i]
        assert s['mask'].min() >= 0.0
        assert s['mask'].max() <= 1.0


def test_tumor_cols_count():
    assert len(TUMOR_COLS) == 9


if __name__ == '__main__':
    tests = [
        test_len,
        test_sample_shapes,
        test_sample_dtypes,
        test_tier1_values,
        test_tier2_values,
        test_tier2_minus1_when_no_tumor,
        test_mask_values,
        test_tumor_cols_count,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)}")