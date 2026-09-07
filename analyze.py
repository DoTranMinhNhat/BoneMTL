# analyze.py
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model   import BoneMTL
from src.dataset import BTXRDDataset, TUMOR_COLS
from src.utils   import load_config, get_device

TUMOR_NAMES = [
    'Osteochondroma', 'Multiple Osteo.', 'Simple Bone Cyst',
    'Giant Cell', 'Osteofibroma', 'Synovial Osteo.',
    'Other Benign', 'Osteosarcoma', 'Other Malig.',
]


def main():
    cfg    = load_config('configs/default.yaml')
    device = get_device()

    # Load model
    model = BoneMTL(num_tumor_types=9, pretrained=False).to(device)
    ckpt  = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load test set
    data_dir = cfg['data']['data_dir']
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_split.csv'))
    tf = A.Compose([
        A.Resize(cfg['data']['img_size'], cfg['data']['img_size']),
        A.Normalize(mean=cfg['data']['mean'], std=cfg['data']['std']),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    loader = DataLoader(
        BTXRDDataset(test_df,
                     os.path.join(data_dir, 'images'),
                     os.path.join(data_dir, 'masks'), tf),
        batch_size=cfg['data']['batch_size'],
        shuffle=False, num_workers=0,
    )

    all_t3_pred, all_t3_lbl = [], []
    all_t1_pred, all_t1_lbl = [], []
    all_t2_pred, all_t2_lbl = [], []

    with torch.no_grad():
        for batch in loader:
            images  = batch['image'].to(device)
            outputs = model(images)

            # Tier 1
            all_t1_pred.append(torch.sigmoid(outputs['tier1']).cpu().numpy().squeeze())
            all_t1_lbl.append(batch['tier1'].numpy().squeeze())

            # Tier 2
            m2 = (batch['tier2'] >= 0).squeeze(1)
            if m2.sum() > 0:
                all_t2_pred.append(
                    torch.sigmoid(outputs['tier2'][m2.to(device)]).cpu().numpy().squeeze()
                )
                all_t2_lbl.append(batch['tier2'][m2].numpy().squeeze())

            # Tier 3 — chỉ tumor samples
            m3 = batch['tier1'].squeeze(1).bool()
            if m3.sum() > 0:
                all_t3_pred.append(
                    torch.softmax(outputs['tier3'][m3.to(device)], dim=1).cpu().numpy()
                )
                all_t3_lbl.append(batch['tier3'][m3].argmax(dim=1).numpy())

    t3_pred = np.concatenate(all_t3_pred).argmax(axis=1)
    t3_lbl  = np.concatenate(all_t3_lbl)

    os.makedirs('results', exist_ok=True)

    # ── 1. Per-class report ────────────────────────────────
    print("\n" + "="*60)
    print("Tier 3 — Per-class Classification Report")
    print("="*60)
    report = classification_report(
        t3_lbl, t3_pred,
        target_names=TUMOR_NAMES,
        digits=4, zero_division=0,
    )
    print(report)

    with open('results/tier3_report.txt', 'w') as f:
        f.write(report)
    print("Saved: results/tier3_report.txt")

    # ── 2. Confusion Matrix Tier 3 ────────────────────────
    cm = confusion_matrix(t3_lbl, t3_pred)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=TUMOR_NAMES,
        yticklabels=TUMOR_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('Tier 3 — Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig('results/confusion_matrix_tier3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/confusion_matrix_tier3.png")

    # ── 3. Per-class F1 bar chart ─────────────────────────
    from sklearn.metrics import f1_score
    f1_per_class = f1_score(t3_lbl, t3_pred, average=None, zero_division=0)

    counts = [int((t3_lbl == i).sum()) for i in range(9)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # F1 per class
    bars = ax1.bar(TUMOR_NAMES, f1_per_class, color='steelblue', edgecolor='navy')
    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title('Tier 3 — Per-class F1 Score', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.axhline(f1_per_class.mean(), color='red', linestyle='--',
                label=f'Macro avg: {f1_per_class.mean():.4f}')
    ax1.legend()
    for bar, val in zip(bars, f1_per_class):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right', fontsize=9)

    # Sample count per class
    bars2 = ax2.bar(TUMOR_NAMES, counts, color='coral', edgecolor='darkred')
    ax2.set_ylabel('Sample Count (Test Set)', fontsize=11)
    ax2.set_title('Tier 3 — Sample Count per Class', fontsize=13, fontweight='bold')
    for bar, val in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=9)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=9)

    plt.tight_layout()
    fig.savefig('results/tier3_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/tier3_per_class.png")

    # ── 4. Confusion Matrix Tier 1 ────────────────────────
    t1_pred_bin = (np.concatenate([np.atleast_1d(x) for x in all_t1_pred]) >= 0.5).astype(int)
    t1_lbl_bin  = np.concatenate([np.atleast_1d(x) for x in all_t1_lbl]).astype(int)

    cm1 = confusion_matrix(t1_lbl_bin, t1_pred_bin)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Greens',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'], ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Ground Truth', fontsize=11)
    ax.set_title('Tier 1 — Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig('results/confusion_matrix_tier1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/confusion_matrix_tier1.png")

    # ── 5. Confusion Matrix Tier 2 ────────────────────────
    t2_pred_bin = (np.concatenate([np.atleast_1d(x) for x in all_t2_pred]) >= 0.5).astype(int)
    t2_lbl_bin  = np.concatenate([np.atleast_1d(x) for x in all_t2_lbl]).astype(int)

    cm2 = confusion_matrix(t2_lbl_bin, t2_pred_bin)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'], ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Ground Truth', fontsize=11)
    ax.set_title('Tier 2 — Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig('results/confusion_matrix_tier2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/confusion_matrix_tier2.png")


if __name__ == '__main__':
    main()