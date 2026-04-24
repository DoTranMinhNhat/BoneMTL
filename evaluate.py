# evaluate.py
import torch

checkpoint = torch.load('checkpoints/best.pth', map_location='cpu')
print(f"Best epoch: {checkpoint['epoch']}")
for k, v in checkpoint['metrics'].items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")