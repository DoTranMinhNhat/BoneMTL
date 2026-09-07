# visualize.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model   import BoneMTL
from src.dataset import BTXRDDataset, TUMOR_COLS
from src.utils   import load_config, get_device

TUMOR_NAMES = [
    'Osteochondroma', 'Multiple Osteo.', 'Simple Bone Cyst',
    'Giant Cell', 'Osteofibroma', 'Synovial Osteo.',
    'Other Benign', 'Osteosarcoma', 'Other Malignant',
]

N_SAMPLES   = 10
OUTPUT_DIR  = 'results'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'visualization.png')


class GradCAM:
    """
    Grad-CAM trên enc4 (ResNet50 stage 4) cho classification head.
    Highlight vùng model chú ý khi predict tier1.

    Reference: Selvaraju et al., Grad-CAM, ICCV 2017
    """

    def __init__(self, model: BoneMTL):
        self.model    = model
        self.gradients = None
        self.activations = None

        # Hook vào enc4 — feature map cuối encoder
        self.hook_a = model.enc4.register_forward_hook(self._save_activation)
        self.hook_g = model.enc4.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, target: str = 'tier1') -> np.ndarray:
        """
        Tính Grad-CAM heatmap cho 1 ảnh.

        Args:
            image_tensor : (1, 3, H, W) tensor trên device
            target       : 'tier1' | 'tier2' | 'tier3'

        Returns:
            heatmap: (H, W) float32 trong [0, 1]
        """
        self.model.zero_grad()
        outputs = self.model(image_tensor)

        # Chọn score để backward
        if target == 'tier1':
            score = outputs['tier1'][0, 0]
        elif target == 'tier2':
            score = outputs['tier2'][0, 0]
        else:
            score = outputs['tier3'][0].max()

        score.backward()

        # Global average pooling trên gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalize về [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def remove_hooks(self):
        self.hook_a.remove()
        self.hook_g.remove()


def load_model(cfg, device):
    model = BoneMTL(num_tumor_types=9, pretrained=False).to(device)
    ckpt  = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def get_samples(cfg, n):
    """Lấy n ảnh mẫu từ test set — ưu tiên ảnh có mask và có tumor."""
    data_dir = cfg['data']['data_dir']
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_split.csv'))

    has_mask = test_df['image_id'].apply(
        lambda x: os.path.exists(
            os.path.join(data_dir, 'masks', x.replace('.jpeg', '_mask.png'))
        )
    )
    pool = test_df[has_mask & (test_df['tumor'] == 1)]
    if len(pool) < n:
        pool = test_df[has_mask]
    if len(pool) < n:
        pool = test_df

    return pool.sample(n=min(n, len(pool)), random_state=42).reset_index(drop=True)


def run_inference(model, image_np, device, cfg):
    """Chạy inference trên 1 ảnh numpy, trả về tensor input và outputs."""
    tf = A.Compose([
        A.Resize(cfg['data']['img_size'], cfg['data']['img_size']),
        A.Normalize(mean=cfg['data']['mean'], std=cfg['data']['std']),
        A.pytorch.ToTensorV2(),
    ])
    tensor = tf(image=image_np)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    return tensor, outputs


def make_overlay(image_np, mask_np, alpha=0.4):
    """Tạo overlay mask đỏ lên ảnh gốc."""
    h, w    = image_np.shape[:2]
    mask_rs = np.array(
        Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
            (w, h), Image.NEAREST
        )
    )
    overlay            = image_np.copy().astype(np.float32)
    red_mask           = mask_rs > 128
    overlay[red_mask, 0] = np.clip(overlay[red_mask, 0] * (1-alpha) + 255*alpha, 0, 255)
    overlay[red_mask, 1] = overlay[red_mask, 1] * (1-alpha)
    overlay[red_mask, 2] = overlay[red_mask, 2] * (1-alpha)
    return overlay.astype(np.uint8)


def apply_gradcam_overlay(image_np, cam, alpha=0.5):
    """Overlay Grad-CAM heatmap lên ảnh gốc."""
    h, w   = image_np.shape[:2]
    cam_rs = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    # Colormap jet
    colormap = plt.cm.jet(cam_rs)[:, :, :3]  # (H, W, 3) RGB
    colormap = (colormap * 255).astype(np.uint8)

    img_f    = image_np.astype(np.float32)
    overlay  = (img_f * (1-alpha) + colormap * alpha).clip(0, 255).astype(np.uint8)
    return overlay


def classification_text(outputs, row):
    """Tạo chuỗi hiển thị kết quả classification."""
    t1_prob = torch.sigmoid(outputs['tier1']).item()
    t1_pred = t1_prob >= 0.5
    t1_gt   = int(row['tumor'])

    t2_prob = torch.sigmoid(outputs['tier2']).item()
    t2_pred = t2_prob >= 0.5
    t2_gt   = int(row['malignant']) if t1_gt == 1 else -1

    t3_prob = torch.softmax(outputs['tier3'], dim=1)[0].cpu().numpy()
    t3_pred = int(t3_prob.argmax())
    t3_gt   = int(pd.Series(row[TUMOR_COLS].values).argmax()) if t1_gt == 1 else -1

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
    return "\n".join(lines)


def main():
    cfg    = load_config('configs/default.yaml')
    device = get_device()
    model  = load_model(cfg, device)
    gradcam = GradCAM(model)

    data_dir = cfg['data']['data_dir']
    img_dir  = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')

    samples  = get_samples(cfg, N_SAMPLES)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Figure: N_SAMPLES hàng × 6 cột
    # Cột: ảnh gốc | GT mask | Pred mask | Overlay | Grad-CAM | Classification
    fig, axes = plt.subplots(
        N_SAMPLES, 6,
        figsize=(24, N_SAMPLES * 4),
        gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1.4]},
    )
    fig.patch.set_facecolor('#0f0f0f')

    col_titles = [
        'X-ray Input', 'GT Mask', 'Pred Mask',
        'Seg Overlay', 'Grad-CAM', 'Classification',
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(
            title, color='white', fontsize=13, fontweight='bold', pad=10
        )

    for i, (_, row) in enumerate(samples.iterrows()):
        img_id   = row['image_id']
        image_np = np.array(
            Image.open(os.path.join(img_dir, img_id)).convert('RGB')
        )

        # Load GT mask
        mask_name = img_id.replace('.jpeg', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            gt_mask = np.array(Image.open(mask_path).convert('L'))
            gt_mask = (gt_mask > 128).astype(np.uint8)
        else:
            gt_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

        # Inference (cần gradient cho Grad-CAM)
        tf = A.Compose([
            A.Resize(cfg['data']['img_size'], cfg['data']['img_size']),
            A.Normalize(mean=cfg['data']['mean'], std=cfg['data']['std']),
            A.pytorch.ToTensorV2(),
        ])
        tensor  = tf(image=image_np)['image'].unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        with torch.no_grad():
            outputs = model(tensor)

        # Grad-CAM — cần enable grad
        tensor_grad = tensor.detach().requires_grad_(True)
        cam = gradcam(tensor_grad, target='tier1')

        # Predicted mask
        pred_logit = outputs['mask'][0, 0].cpu().numpy()
        pred_prob  = 1 / (1 + np.exp(-pred_logit))
        pred_mask  = (pred_prob > 0.5).astype(np.uint8)

        h, w = image_np.shape[:2]
        pred_mask_rs = np.array(
            Image.fromarray(pred_mask * 255).resize((w, h), Image.NEAREST)
        )
        gt_mask_rs = np.array(
            Image.fromarray(gt_mask * 255).resize((w, h), Image.NEAREST)
        )

        # Overlay segmentation
        seg_overlay  = make_overlay(image_np, pred_mask_rs / 255.0)
        # Overlay Grad-CAM
        cam_overlay  = apply_gradcam_overlay(image_np, cam)

        # Plot
        axes[i, 0].imshow(image_np, cmap='gray')
        axes[i, 0].set_ylabel(
            img_id[:16], color='#aaaaaa', fontsize=8,
            rotation=0, labelpad=80, va='center',
        )
        axes[i, 1].imshow(gt_mask_rs,   cmap='Blues', vmin=0, vmax=255)
        axes[i, 2].imshow(pred_mask_rs, cmap='Reds',  vmin=0, vmax=255)
        axes[i, 3].imshow(seg_overlay)
        axes[i, 4].imshow(cam_overlay)

        # Classification text
        t1_correct = (
            (torch.sigmoid(outputs['tier1']).item() >= 0.5) == bool(row['tumor'])
        )
        text_color = '#00ff88' if t1_correct else '#ff6666'
        axes[i, 5].set_facecolor('#1a1a1a')
        axes[i, 5].text(
            0.05, 0.5, classification_text(outputs, row),
            transform=axes[i, 5].transAxes,
            color=text_color, fontsize=10,
            va='center', fontfamily='monospace',
        )
        axes[i, 5].axis('off')

        for j in range(5):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor('#333333')

    legend_elements = [
        mpatches.Patch(color='#00ff88', label='Tier 1 correct'),
        mpatches.Patch(color='#ff6666', label='Tier 1 incorrect'),
    ]
    fig.legend(
        handles=legend_elements, loc='lower center',
        ncol=2, fontsize=11, facecolor='#1a1a1a',
        labelcolor='white', framealpha=0.8,
        bbox_to_anchor=(0.5, 0.005),
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
    plt.close()
    gradcam.remove_hooks()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()