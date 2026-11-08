import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import TUMOR_NAMES, BTXRDDataset
from src.inference import load_model
from src.transforms import build_transforms
from src.utils import get_device, load_config

N_SAMPLES = 10
OUTPUT_DIR = 'results'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'visualization.png')

COLUMN_TITLES = [
    'X-ray Input', 'GT Mask', 'Pred Mask',
    'Seg Overlay', 'Grad-CAM', 'Classification',
]


class GradCAM:
    """
    Grad-CAM on enc4 (last ResNet50 stage) for a classification output.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = [
            model.enc4.register_forward_hook(self._save_activation),
            model.enc4.register_full_backward_hook(self._save_gradient),
        ]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for hook in self._hooks:
            hook.remove()

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, target: str = 'tier1'):
        """
        image is (1, 3, H, W) on the model device and target is 'tier1', 'tier2' or 'tier3'.
        Returns the heatmap in [0, 1] at feature-map resolution and the detached model outputs.
        """

        self.model.zero_grad()
        outputs = self.model(image.detach().requires_grad_(True))

        if target == 'tier3':
            score = outputs['tier3'][0].max()
        else:
            score = outputs[target][0, 0]
        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, {k: v.detach() for k, v in outputs.items()}


def mask_path_for(mask_dir: str, img_id: str) -> str:
    return os.path.join(mask_dir, os.path.splitext(img_id)[0] + '_mask.png')


def get_samples(cfg: dict, n: int) -> pd.DataFrame:
    data_dir = cfg['data']['data_dir']
    mask_dir = os.path.join(data_dir, 'masks')
    test_df = pd.read_csv(os.path.join(data_dir, 'test_split.csv'))

    has_mask = test_df['image_id'].apply(
        lambda x: os.path.exists(mask_path_for(mask_dir, x))
    )
    pool = test_df[has_mask & (test_df['tumor'] == 1)]
    if len(pool) < n:
        pool = test_df[has_mask]
    if len(pool) < n:
        pool = test_df

    return pool.sample(n=min(n, len(pool)), random_state=42).reset_index(drop=True)


def load_gt_mask(path: str, size: tuple) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros((size[1], size[0]), dtype=bool)
    mask = Image.open(path).convert('L').resize(size, Image.NEAREST)
    return np.array(mask) > 128


def upsample_mask(mask: np.ndarray, size: tuple) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        size, Image.NEAREST
    )
    return np.array(resized) > 0


def make_overlay(image: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.4) -> np.ndarray:
    overlay = image.astype(np.float32)
    overlay[mask, 0] = np.clip(
        overlay[mask, 0] * (1 - alpha) + 255 * alpha, 0, 255
    )
    overlay[mask, 1] *= 1 - alpha
    overlay[mask, 2] *= 1 - alpha
    return overlay.astype(np.uint8)


def apply_gradcam_overlay(image: np.ndarray, cam: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    h, w = image.shape[:2]
    cam_rs = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    heat = (plt.cm.jet(cam_rs)[:, :, :3] * 255).astype(np.uint8)
    blended = image.astype(np.float32) * (1 - alpha) + heat * alpha
    return blended.clip(0, 255).astype(np.uint8)


def describe_prediction(outputs: dict, sample: dict):
    t1_prob = torch.sigmoid(outputs['tier1']).item()
    t1_pred = t1_prob >= 0.5
    t1_gt = int(sample['tier1'].item())

    t2_prob = torch.sigmoid(outputs['tier2']).item()
    t2_pred = t2_prob >= 0.5
    t2_gt = int(sample['tier2'].item())

    t3_pred = int(outputs['tier3'][0].argmax())
    t3_gt = int(sample['tier3'].argmax()) if t1_gt == 1 else -1

    lines = [
        f"T1 pred: {'Tumor' if t1_pred else 'Normal'} ({t1_prob:.0%})",
        f"T1 GT:   {'Tumor' if t1_gt else 'Normal'}",
        "",
        f"T2 pred: {'Malig.' if t2_pred else 'Benign'} ({t2_prob:.0%})"
        if t1_pred else "T2 pred: N/A",
        f"T2 GT:   {'Malig.' if t2_gt == 1 else 'Benign'}"
        if t2_gt >= 0 else "T2 GT:   N/A",
        "",
        f"T3 pred: {TUMOR_NAMES[t3_pred]}",
        f"T3 GT:   {TUMOR_NAMES[t3_gt]}" if t3_gt >= 0 else "T3 GT:   N/A",
    ]
    return "\n".join(lines), t1_pred == bool(t1_gt)


def plot_row(axes, img_id, image, gt_mask, pred_mask, cam, text, correct):
    axes[0].imshow(image)
    axes[0].set_ylabel(
        img_id[:16], color='#aaaaaa', fontsize=8,
        rotation=0, labelpad=80, va='center',
    )
    axes[1].imshow(gt_mask, cmap='Blues', vmin=0, vmax=1)
    axes[2].imshow(pred_mask, cmap='Reds', vmin=0, vmax=1)
    axes[3].imshow(make_overlay(image, pred_mask))
    axes[4].imshow(apply_gradcam_overlay(image, cam))

    axes[5].set_facecolor('#1a1a1a')
    axes[5].text(
        0.05, 0.5, text, transform=axes[5].transAxes,
        color='#00ff88' if correct else '#ff6666', fontsize=10,
        va='center', fontfamily='monospace',
    )
    axes[5].axis('off')

    for ax in axes[:5]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')


def main():
    cfg = load_config('configs/default.yaml')
    device = get_device()
    model = load_model(
        os.path.join(cfg['checkpoint']['save_dir'], 'swa.pth'), cfg, device
    )

    data_dir = cfg['data']['data_dir']
    img_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')

    samples = get_samples(cfg, N_SAMPLES)
    _, val_tf = build_transforms(cfg)
    dataset = BTXRDDataset(samples, img_dir, mask_dir, val_tf)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        N_SAMPLES, 6,
        figsize=(24, N_SAMPLES * 4),
        gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1.4]},
    )
    fig.patch.set_facecolor('#0f0f0f')
    for j, title in enumerate(COLUMN_TITLES):
        axes[0, j].set_title(
            title, color='white', fontsize=13, fontweight='bold', pad=10
        )

    with GradCAM(model) as gradcam:
        for i, img_id in enumerate(samples['image_id']):
            sample = dataset[i]
            image = np.array(
                Image.open(os.path.join(img_dir, img_id)).convert('RGB')
            )
            size = (image.shape[1], image.shape[0])

            cam, outputs = gradcam(sample['image'].unsqueeze(0).to(device))

            pred_mask = (torch.sigmoid(outputs['mask'][0, 0]) > 0.5)
            pred_mask = upsample_mask(pred_mask.cpu().numpy(), size)
            gt_mask = load_gt_mask(mask_path_for(mask_dir, img_id), size)

            text, correct = describe_prediction(outputs, sample)
            plot_row(axes[i], img_id, image, gt_mask, pred_mask, cam,
                     text, correct)

    fig.legend(
        handles=[
            mpatches.Patch(color='#00ff88', label='Tier 1 correct'),
            mpatches.Patch(color='#ff6666', label='Tier 1 incorrect'),
        ],
        loc='lower center', ncol=2, fontsize=11, facecolor='#1a1a1a',
        labelcolor='white', framealpha=0.8, bbox_to_anchor=(0.5, 0.005),
    )
    plt.suptitle(
        'BoneMTL — Bone Tumor Classification & Segmentation Results',
        color='white', fontsize=16, fontweight='bold', y=1.002,
    )
    plt.tight_layout(pad=1.5)
    fig.savefig(
        OUTPUT_PATH, dpi=150, bbox_inches='tight',
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()