import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

TUMOR_COLS = [
    'osteochondroma', 'multiple osteochondromas', 'simple bone cyst',
    'giant cell tumor', 'osteofibroma', 'synovial osteochondroma',
    'other bt', 'osteosarcoma', 'other mt',
]

TUMOR_NAMES = [
    'Osteochondroma', 'Multiple Osteo.', 'Simple Bone Cyst',
    'Giant Cell', 'Osteofibroma', 'Synovial Osteo.',
    'Other Benign', 'Osteosarcoma', 'Other Malig.',
]

MAX_LOAD_SIZE = 1024


class BTXRDDataset(Dataset):
    """
    BTXRD bone tumor X-ray dataset.

    Each sample is a dict:
        image    : (3, H, W) float32, normalized
        mask     : (1, H, W) float32, values 0.0 or 1.0
        tier1    : (1,) tumor=1 / no_tumor=0
        tier2    : (1,) malignant=1 / benign=0 / -1 if no tumor
        tier3    : (9,) one-hot tumor type
        has_mask : bool, whether a pixel-level mask exists
        image_id : str, source file name
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        mask_dir: str,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        try:
            img = Image.open(path).convert('RGB')
        except OSError as e:
            raise OSError(f"Cannot read image {path}: {e}") from e
        if max(img.size) > MAX_LOAD_SIZE:
            ratio = MAX_LOAD_SIZE / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.BILINEAR)
        return np.array(img)

    @staticmethod
    def _load_mask(path: str, ref_shape: tuple) -> np.ndarray:
        mask = Image.open(path).convert('L')
        if mask.size != (ref_shape[1], ref_shape[0]):
            mask = mask.resize((ref_shape[1], ref_shape[0]), Image.NEAREST)
        return (np.array(mask) > 128).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_id = row['image_id']

        image = self._load_image(os.path.join(self.img_dir, img_id))

        mask_name = os.path.splitext(img_id)[0] + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        if os.path.exists(mask_path):
            mask = self._load_mask(mask_path, image.shape[:2])
            has_mask = True
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            has_mask = False

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out['image'], out['mask']

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).float()

        tier1 = torch.tensor([float(row['tumor'])], dtype=torch.float32)
        tier2 = torch.tensor(
            [float(row['malignant'])] if row['tumor'] == 1 else [-1.0],
            dtype=torch.float32,
        )
        tier3 = torch.tensor(
            row[TUMOR_COLS].values.astype(np.float32), dtype=torch.float32
        )

        return {
            'image': image,
            'mask': mask,
            'tier1': tier1,
            'tier2': tier2,
            'tier3': tier3,
            'has_mask': torch.tensor(has_mask, dtype=torch.bool),
            'image_id': img_id,
        }