import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.dataset import TUMOR_NAMES
from src.inference import (
    Collector, build_test_loader, concat, load_model, predict,
)
from src.utils import get_device, load_config

LABELS = list(range(len(TUMOR_NAMES)))


def save_report(t3_lbl, t3_pred, path: str):
    report = classification_report(
        t3_lbl, t3_pred, labels=LABELS, target_names=TUMOR_NAMES,
        digits=4, zero_division=0,
    )
    print(f"\n{'=' * 60}\nTier 3: per-class classification report\n{'=' * 60}")
    print(report)
    with open(path, 'w') as f:
        f.write(report)
    print(f"Saved: {path}")


def plot_confusion_matrix(cm, labels, title, path, cmap, figsize,
                          many_classes: bool = False):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap,
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5 if many_classes else 0,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if many_classes:
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_class(f1_per_class, counts, path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    bars = ax1.bar(TUMOR_NAMES, f1_per_class,
                   color='steelblue', edgecolor='navy')
    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title('Tier 3 — Per-class F1 Score', fontsize=13,
                  fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.axhline(f1_per_class.mean(), color='red', linestyle='--',
                label=f'Macro avg: {f1_per_class.mean():.4f}')
    ax1.legend()
    for bar, val in zip(bars, f1_per_class):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(TUMOR_NAMES, counts, color='coral', edgecolor='darkred')
    ax2.set_ylabel('Sample Count (Test Set)', fontsize=11)
    ax2.set_title('Tier 3 — Sample Count per Class', fontsize=13,
                  fontweight='bold')
    for bar, val in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=9)

    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    cfg = load_config('configs/default.yaml')
    device = get_device()

    model = load_model(
        os.path.join(cfg['checkpoint']['save_dir'], 'swa.pth'), cfg, device
    )

    collector = Collector()
    for batch in build_test_loader(cfg):
        pred = predict(model, batch['image'].to(device))
        collector.add_cls(pred, batch, device)

    os.makedirs('results', exist_ok=True)

    t3_pred = np.concatenate(collector.t3_pred).argmax(axis=1)
    t3_lbl = np.concatenate(collector.t3_lbl)

    save_report(t3_lbl, t3_pred, 'results/tier3_report.txt')

    plot_confusion_matrix(
        confusion_matrix(t3_lbl, t3_pred, labels=LABELS), TUMOR_NAMES,
        'Tier 3 — Confusion Matrix (Test Set)',
        'results/confusion_matrix_tier3.png',
        cmap='Blues', figsize=(11, 9), many_classes=True,
    )

    f1_per_class = f1_score(
        t3_lbl, t3_pred, labels=LABELS, average=None, zero_division=0
    )
    counts = [int((t3_lbl == i).sum()) for i in LABELS]
    plot_per_class(f1_per_class, counts, 'results/tier3_per_class.png')

    t1_pred = (concat(collector.t1_pred) >= 0.5).astype(int)
    t1_lbl = concat(collector.t1_lbl).astype(int)
    plot_confusion_matrix(
        confusion_matrix(t1_lbl, t1_pred), ['No Tumor', 'Tumor'],
        'Tier 1 — Confusion Matrix', 'results/confusion_matrix_tier1.png',
        cmap='Greens', figsize=(5, 4),
    )

    t2_pred = (concat(collector.t2_pred) >= 0.5).astype(int)
    t2_lbl = concat(collector.t2_lbl).astype(int)
    plot_confusion_matrix(
        confusion_matrix(t2_lbl, t2_pred), ['Benign', 'Malignant'],
        'Tier 2 — Confusion Matrix', 'results/confusion_matrix_tier2.png',
        cmap='Oranges', figsize=(5, 4),
    )


if __name__ == '__main__':
    main()