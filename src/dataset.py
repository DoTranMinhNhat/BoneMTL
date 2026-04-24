# src/dataset.py
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# Các loại u xương trong dataset BTXRD
TUMOR_COLS = [
    'osteochondroma', 'multiple osteochondromas', 'simple bone cyst',
    'giant cell tumor', 'osteofibroma', 'synovial osteochondroma',
    'other bt', 'osteosarcoma', 'other mt',
]


class BTXRDDataset(Dataset):
    """
    PyTorch Dataset cho bone tumor X-ray dataset.

    Mỗi sample trả về dict:
        image    : (3, H, W)  float32 normalized
        mask     : (1, H, W)  float32  0.0 | 1.0
        tier1    : (1,)       float32  tumor=1 / no_tumor=0
        tier2    : (1,)       float32  malignant=1 / benign=0 / -1=N/A
        tier3    : (9,)       float32  one-hot tumor type
        has_mask : bool       ảnh có pixel-level mask hay không
        image_id : str        tên file gốc

    Args:
        df        : DataFrame đã split (train/val/test)
        img_dir   : thư mục chứa ảnh .jpeg
        mask_dir  : thư mục chứa mask .png
        transform : albumentations Compose
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        img_dir:   str,
        mask_dir:  str,
        transform = None,
    ):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row    = self.df.iloc[idx]
        img_id = row['image_id']

        # Load ảnh gốc
        image = np.array(
            Image.open(os.path.join(self.img_dir, img_id)).convert('RGB')
        )

        # Load mask định dạng PNG
        mask_name = img_id.replace('.jpeg', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        if os.path.exists(mask_path):
            mask     = np.array(Image.open(mask_path).convert('L'))
            mask     = (mask > 128).astype(np.float32)
            has_mask = True
        else:
            # Ảnh không có u, tạo mask rỗng
            mask     = np.zeros(image.shape[:2], dtype=np.float32)
            has_mask = False

        # Áp dụng augmentation
        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out['image']
            mask  = out['mask']

        # Chuyển mask sang (1, H, W)
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Nhãn phân cấp 3 tầng
        tier1 = torch.tensor([float(row['tumor'])], dtype=torch.float32)
        tier2 = torch.tensor(
            [float(row['malignant'])] if row['tumor'] == 1 else [-1.0],
            dtype=torch.float32,
        )
        tier3 = torch.tensor(
            row[TUMOR_COLS].values.astype(np.float32),
            dtype=torch.float32,
        )

        return {
            'image':    image,
            'mask':     mask,
            'tier1':    tier1,
            'tier2':    tier2,
            'tier3':    tier3,
            'has_mask': torch.tensor(has_mask, dtype=torch.bool),
            'image_id': img_id,
        }