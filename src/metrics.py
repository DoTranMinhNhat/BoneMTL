# src/metrics.py
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_dice(
    pred: torch.Tensor, target: torch.Tensor,
    threshold: float = 0.5, smooth: float = 1e-6,
) -> float:
    """Dice coefficient cho binary segmentation."""
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter  = (pred * target).sum(dim=1)
    dice   = (2.0 * inter + smooth) / (
        pred.sum(dim=1) + target.sum(dim=1) + smooth
    )
    return dice.mean().item()


def compute_iou(
    pred: torch.Tensor, target: torch.Tensor,
    threshold: float = 0.5, smooth: float = 1e-6,
) -> float:
    """IoU (Intersection over Union) cho binary segmentation."""
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter  = (pred * target).sum(dim=1)
    union  = pred.sum(dim=1) + target.sum(dim=1) - inter
    iou    = (inter + smooth) / (union + smooth)
    return iou.mean().item()


def compute_cls_metrics(
    preds: np.ndarray, labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Accuracy, F1, AUC cho binary classification."""
    preds_bin = (preds >= threshold).astype(int)
    accuracy  = (preds_bin == labels).mean()
    f1        = f1_score(labels, preds_bin, zero_division=0)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.0
    return {'accuracy': float(accuracy), 'f1': float(f1), 'auc': float(auc)}


def compute_tier3_metrics(
    preds: np.ndarray, labels: np.ndarray,
) -> dict:
    """Accuracy, F1 macro/weighted cho 9-class classification."""
    pred_cls    = preds.argmax(axis=1)
    accuracy    = (pred_cls == labels).mean()
    f1_macro    = f1_score(labels, pred_cls, average='macro',    zero_division=0)
    f1_weighted = f1_score(labels, pred_cls, average='weighted', zero_division=0)
    return {
        'accuracy':    float(accuracy),
        'f1_macro':    float(f1_macro),
        'f1_weighted': float(f1_weighted),
    }


class MetricTracker:
    """Tích lũy và tính average metrics qua nhiều batch."""

    def __init__(self):
        self.data = {}

    def reset(self):
        self.data = {}

    def update(self, key: str, value: float, n: int = 1):
        if key not in self.data:
            self.data[key] = {'sum': 0.0, 'count': 0}
        self.data[key]['sum']   += value * n
        self.data[key]['count'] += n

    def result(self) -> dict:
        return {k: v['sum'] / v['count']
                for k, v in self.data.items() if v['count'] > 0}

    def __str__(self) -> str:
        return ' | '.join(f"{k}: {v:.4f}" for k, v in self.result().items())