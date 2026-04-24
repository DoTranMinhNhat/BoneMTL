# src/utils.py
import os
import yaml
import torch
import random
import numpy as np


def load_config(config_path: str) -> dict:
    """Đọc YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Sửa tất cả random seed để kết quả reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Trả về GPU nếu có, CPU nếu không."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    return device


def save_checkpoint(state: dict, save_dir: str, filename: str):
    """Lưu model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))


def load_checkpoint(path: str, model, optimizer=None):
    """Load checkpoint. Trả về (epoch, metrics)."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})