import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import BTXRDDataset
from src.metrics import compute_cls_metrics, compute_tier3_metrics
from src.model import BoneMTL
from src.transforms import build_transforms
from src.utils import load_checkpoint

# (forward, inverse) pairs on (B, C, H, W) tensors
TTA_VIEWS = (
    (lambda x: x, lambda x: x),
    (lambda x: x.flip(3), lambda x: x.flip(3)),
    (lambda x: x.flip(2), lambda x: x.flip(2)),
    (lambda x: torch.rot90(x, 1, (2, 3)), lambda x: torch.rot90(x, -1, (2, 3))),
    (lambda x: x.flip((2, 3)), lambda x: x.flip((2, 3))),
)


def concat(parts) -> np.ndarray:
    return np.concatenate([np.atleast_1d(x.flatten()) for x in parts])


@torch.no_grad()
def predict(model, images, tta: bool = False) -> dict:
    views = TTA_VIEWS if tta else TTA_VIEWS[:1]
    t1 = t2 = t3 = mask = 0
    for forward, inverse in views:
        out = model(forward(images))
        t1 = t1 + torch.sigmoid(out['tier1'])
        t2 = t2 + torch.sigmoid(out['tier2'])
        t3 = t3 + torch.softmax(out['tier3'], dim=1)
        mask = mask + inverse(out['mask'])
    n = len(views)
    return {'tier1': t1 / n, 'tier2': t2 / n, 'tier3': t3 / n, 'mask': mask / n}


class Collector:
    def __init__(self):
        self.t1_pred, self.t1_lbl = [], []
        self.t2_pred, self.t2_lbl = [], []
        self.t3_pred, self.t3_lbl = [], []
        self.dice, self.iou = [], []

    def add_cls(self, pred: dict, batch: dict, device):
        self.t1_pred.append(pred['tier1'].cpu().numpy().squeeze())
        self.t1_lbl.append(batch['tier1'].numpy().squeeze())

        m2 = (batch['tier2'] >= 0).squeeze(1)
        if m2.sum() > 0:
            self.t2_pred.append(
                pred['tier2'][m2.to(device)].cpu().numpy().squeeze()
            )
            self.t2_lbl.append(batch['tier2'][m2].numpy().squeeze())

        m3 = batch['tier1'].squeeze(1).bool()
        if m3.sum() > 0:
            self.t3_pred.append(pred['tier3'][m3.to(device)].cpu().numpy())
            self.t3_lbl.append(batch['tier3'][m3].argmax(dim=1).numpy())

    def add_seg(self, dice: float, iou: float):
        self.dice.append(dice)
        self.iou.append(iou)

    def result(self) -> dict:
        result = {}

        if self.t1_pred:
            m = compute_cls_metrics(concat(self.t1_pred), concat(self.t1_lbl))
            result.update({
                'tier1_acc': m['accuracy'],
                'tier1_f1': m['f1'],
                'tier1_auc': m['auc'],
            })

        if self.t2_pred:
            m = compute_cls_metrics(
                concat(self.t2_pred), concat(self.t2_lbl).astype(int)
            )
            result.update({
                'tier2_acc': m['accuracy'],
                'tier2_f1': m['f1'],
                'tier2_auc': m['auc'],
            })

        if self.t3_pred:
            m = compute_tier3_metrics(
                np.concatenate(self.t3_pred, axis=0),
                np.concatenate(self.t3_lbl, axis=0),
            )
            result.update({
                'tier3_acc': m['accuracy'],
                'tier3_f1_macro': m['f1_macro'],
            })

        if self.dice:
            result['dice'] = float(np.mean(self.dice))
            result['iou'] = float(np.mean(self.iou))

        return result


def load_model(path: str, cfg: dict, device) -> BoneMTL:
    model = BoneMTL(
        num_tumor_types=cfg['model']['num_classes'],
        pretrained=False,
    ).to(device)
    epoch, _ = load_checkpoint(path, model)
    print(f"Loaded {os.path.basename(path)} (epoch {epoch})")
    return model.eval()


def build_test_loader(cfg: dict) -> DataLoader:
    data_dir = cfg['data']['data_dir']
    test_df = pd.read_csv(os.path.join(data_dir, 'test_split.csv'))
    _, test_tf = build_transforms(cfg)
    return DataLoader(
        BTXRDDataset(
            test_df,
            os.path.join(data_dir, 'images'),
            os.path.join(data_dir, 'masks'),
            test_tf,
        ),
        batch_size=cfg['data']['batch_size'],
        shuffle=False, num_workers=0,
    )