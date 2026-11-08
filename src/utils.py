import os
import random

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {vram:.1f} GB")
    return device


def save_checkpoint(state: dict, save_dir: str, filename: str):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})