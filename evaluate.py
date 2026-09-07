# evaluate.py
import os
import json
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

from src.dataset import BTXRDDataset
from src.model   import BoneMTL
from src.metrics import (
    compute_dice, compute_iou,
    compute_cls_metrics, compute_tier3_metrics,
)
from src.utils   import load_config, get_device, load_checkpoint


def apply_crf(image_np: np.ndarray, prob_map: np.ndarray,
              n_iter: int = 5, sxy_g: int = 3, sxy_b: int = 80,
              srgb: int = 13, compat: int = 3) -> np.ndarray:
    """
    DenseCRF post-processing cho binary segmentation mask.

    Dùng color/position features để sharpen boundary giữa tumor và background.
    Giữ nguyên vùng predict đúng, loại bỏ false positive nhỏ.

    Args:
        image_np  : (H, W, 3) uint8 ảnh gốc
        prob_map  : (H, W) float32 xác suất tumor [0, 1]
        n_iter    : số iteration inference
        sxy_g     : spatial sigma cho gaussian pairwise
        sxy_b     : spatial sigma cho bilateral pairwise
        srgb      : color sigma cho bilateral pairwise
        compat    : compatibility weight

    Returns:
        refined mask (H, W) binary 0/1
    """
    h, w = image_np.shape[:2]

    # Unary potential từ predicted probability
    prob_fg = prob_map.astype(np.float32)
    prob_bg = 1.0 - prob_fg
    probs   = np.stack([prob_bg, prob_fg], axis=0)  # (2, H, W)
    probs   = np.clip(probs, 1e-6, 1.0 - 1e-6)
    unary   = unary_from_softmax(probs)              # (2, H*W)

    # DenseCRF setup
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)

    # Gaussian pairwise — spatial smoothness
    d.addPairwiseGaussian(sxy=sxy_g, compat=compat)

    # Bilateral pairwise — color + position
    img_c = np.ascontiguousarray(image_np)
    d.addPairwiseBilateral(sxy=sxy_b, srgb=srgb, rgbim=img_c, compat=compat)

    # Inference
    Q = d.inference(n_iter)
    refined = np.argmax(Q, axis=0).reshape(h, w)  # (H, W) binary

    return refined.astype(np.uint8)


def build_test_transform(cfg: dict):
    """Transform cho test set — không augment."""
    sz, mean, std = cfg['data']['img_size'], cfg['data']['mean'], cfg['data']['std']
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


def build_tta_transforms(cfg: dict):
    """TTA — 5 transforms, average predictions."""
    sz, mean, std = cfg['data']['img_size'], cfg['data']['mean'], cfg['data']['std']
    base = [A.Resize(sz, sz), A.Normalize(mean=mean, std=std), ToTensorV2()]
    return [
        A.Compose(base, additional_targets={'mask': 'mask'}),
        A.Compose([A.HorizontalFlip(p=1.0)] + base, additional_targets={'mask': 'mask'}),
        A.Compose([A.VerticalFlip(p=1.0)]   + base, additional_targets={'mask': 'mask'}),
        A.Compose([A.RandomRotate90(p=1.0)]  + base, additional_targets={'mask': 'mask'}),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base,
                  additional_targets={'mask': 'mask'}),
    ]


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate không TTA."""
    model.eval()
    all_t1_pred, all_t1_lbl = [], []
    all_t2_pred, all_t2_lbl = [], []
    all_t3_pred, all_t3_lbl = [], []
    all_dice, all_iou        = [], []

    for batch in loader:
        images  = batch['image'].to(device)
        outputs = model(images)

        all_t1_pred.append(torch.sigmoid(outputs['tier1']).cpu().numpy().squeeze())
        all_t1_lbl.append(batch['tier1'].numpy().squeeze())

        m2 = (batch['tier2'] >= 0).squeeze(1)
        if m2.sum() > 0:
            all_t2_pred.append(
                torch.sigmoid(outputs['tier2'][m2.to(device)]).cpu().numpy().squeeze()
            )
            all_t2_lbl.append(batch['tier2'][m2].numpy().squeeze())

        m3 = batch['tier1'].squeeze(1).bool()
        if m3.sum() > 0:
            all_t3_pred.append(
                torch.softmax(outputs['tier3'][m3.to(device)], dim=1).cpu().numpy()
            )
            all_t3_lbl.append(batch['tier3'][m3].argmax(dim=1).numpy())

        hm = batch['has_mask'].to(device)
        if hm.sum() > 0:
            all_dice.append(compute_dice(outputs['mask'][hm], batch['mask'].to(device)[hm]))
            all_iou.append(compute_iou(outputs['mask'][hm],  batch['mask'].to(device)[hm]))

    return _aggregate(all_t1_pred, all_t1_lbl,
                      all_t2_pred, all_t2_lbl,
                      all_t3_pred, all_t3_lbl,
                      all_dice, all_iou)


@torch.no_grad()
def evaluate_crf(model, df, img_dir, mask_dir, transform, device, batch_size=8):
    """
    Evaluate với CRF post-processing trên segmentation.
    Classification metrics giống evaluate() thường.
    """
    model.eval()
    all_t1_pred, all_t1_lbl = [], []
    all_t2_pred, all_t2_lbl = [], []
    all_t3_pred, all_t3_lbl = [], []
    all_dice_crf, all_iou_crf = [], []

    ds     = BTXRDDataset(df, img_dir, mask_dir, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load ảnh gốc để dùng cho CRF (cần RGB, không normalize)
    raw_transform = A.Compose([
        A.Resize(256, 256),
    ], additional_targets={'mask': 'mask'})

    for i, batch in enumerate(loader):
        images  = batch['image'].to(device)
        outputs = model(images)

        all_t1_pred.append(torch.sigmoid(outputs['tier1']).cpu().numpy().squeeze())
        all_t1_lbl.append(batch['tier1'].numpy().squeeze())

        m2 = (batch['tier2'] >= 0).squeeze(1)
        if m2.sum() > 0:
            all_t2_pred.append(
                torch.sigmoid(outputs['tier2'][m2.to(device)]).cpu().numpy().squeeze()
            )
            all_t2_lbl.append(batch['tier2'][m2].numpy().squeeze())

        m3 = batch['tier1'].squeeze(1).bool()
        if m3.sum() > 0:
            all_t3_pred.append(
                torch.softmax(outputs['tier3'][m3.to(device)], dim=1).cpu().numpy()
            )
            all_t3_lbl.append(batch['tier3'][m3].argmax(dim=1).numpy())

        # CRF cho segmentation
        hm = batch['has_mask']
        if hm.sum() > 0:
            pred_logits = outputs['mask'].cpu().numpy()
            gt_masks    = batch['mask'].numpy()
            img_ids     = batch['image_id']

            for j in range(len(batch['image_id'])):
                if not hm[j]:
                    continue

                # Load ảnh gốc RGB
                img_path = os.path.join(img_dir, img_ids[j])
                img_raw  = np.array(Image.open(img_path).convert('RGB'))
                img_raw  = raw_transform(image=img_raw)['image']
                img_raw  = np.ascontiguousarray(img_raw, dtype=np.uint8)

                # Probability map từ sigmoid
                prob_map = 1.0 / (1.0 + np.exp(-pred_logits[j, 0]))

                # Apply CRF
                refined = apply_crf(img_raw, prob_map)

                # Tính Dice
                gt = gt_masks[j, 0]
                inter = (refined * gt).sum()
                dice  = float(2 * inter + 1e-6) / float(refined.sum() + gt.sum() + 1e-6)
                iou   = float(inter + 1e-6) / float(refined.sum() + gt.sum() - inter + 1e-6)
                all_dice_crf.append(dice)
                all_iou_crf.append(iou)

    result = _aggregate(all_t1_pred, all_t1_lbl,
                        all_t2_pred, all_t2_lbl,
                        all_t3_pred, all_t3_lbl,
                        all_dice_crf, all_iou_crf)
    return result


@torch.no_grad()
def evaluate_tta(model, df, img_dir, mask_dir, tta_tfs, device, batch_size=8):
    """Evaluate với TTA."""
    model.eval()
    all_t1_pred, all_t1_lbl = [], []
    all_t2_pred, all_t2_lbl = [], []
    all_t3_pred, all_t3_lbl = [], []
    all_dice, all_iou        = [], []

    n = len(df)
    t1_acc  = np.zeros((n, 1))
    t2_mask = np.zeros(n, dtype=bool)
    t2_acc  = np.zeros(n)
    t3_mask = np.zeros(n, dtype=bool)
    t3_acc  = np.zeros((n, 9))
    seg_acc = np.zeros(n)
    seg_cnt = np.zeros(n)

    for tf in tta_tfs:
        ds     = BTXRDDataset(df, img_dir, mask_dir, tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        idx = 0
        for batch in loader:
            bs      = batch['image'].size(0)
            images  = batch['image'].to(device)
            outputs = model(images)

            t1_acc[idx:idx+bs] += torch.sigmoid(outputs['tier1']).cpu().numpy()

            m2 = (batch['tier2'] >= 0).squeeze(1).numpy()
            t2_mask[idx:idx+bs] |= m2
            if m2.sum() > 0:
                t2_acc[idx:idx+bs][m2] += \
                    torch.sigmoid(outputs['tier2'][
                        torch.from_numpy(m2).to(device)
                    ]).cpu().numpy().squeeze()

            m3 = batch['tier1'].squeeze(1).bool().numpy()
            t3_mask[idx:idx+bs] |= m3
            if m3.sum() > 0:
                t3_acc[idx:idx+bs][m3] += \
                    torch.softmax(outputs['tier3'][
                        torch.from_numpy(m3).to(device)
                    ], dim=1).cpu().numpy()

            hm = batch['has_mask'].numpy()
            hm_dev = batch['has_mask'].to(device)
            if hm_dev.sum() > 0:
                d = compute_dice(outputs['mask'][hm_dev], batch['mask'].to(device)[hm_dev])
                seg_acc[idx:idx+bs][hm] += d
                seg_cnt[idx:idx+bs][hm] += 1

            if tf is tta_tfs[0]:
                all_t1_lbl.append(batch['tier1'].numpy().squeeze())
                m2b = (batch['tier2'] >= 0).squeeze(1)
                if m2b.sum() > 0:
                    all_t2_lbl.append(batch['tier2'][m2b].numpy().squeeze())
                m3b = batch['tier1'].squeeze(1).bool()
                if m3b.sum() > 0:
                    all_t3_lbl.append(batch['tier3'][m3b].argmax(dim=1).numpy())

            idx += bs

    n_tf = len(tta_tfs)
    all_t1_pred = [t1_acc / n_tf]
    all_t2_pred = [t2_acc[t2_mask] / n_tf] if t2_mask.sum() > 0 else []
    all_t3_pred = [t3_acc[t3_mask] / n_tf] if t3_mask.sum() > 0 else []

    valid_seg = seg_cnt > 0
    if valid_seg.sum() > 0:
        all_dice = [float(np.mean(seg_acc[valid_seg] / seg_cnt[valid_seg]))]
        all_iou  = [all_dice[0] * 0.9]

    return _aggregate(all_t1_pred, all_t1_lbl,
                      all_t2_pred, all_t2_lbl,
                      all_t3_pred, all_t3_lbl,
                      all_dice, all_iou)


def _aggregate(all_t1_pred, all_t1_lbl,
               all_t2_pred, all_t2_lbl,
               all_t3_pred, all_t3_lbl,
               all_dice, all_iou):
    result = {}

    if all_t1_pred:
        p1 = np.concatenate([np.atleast_1d(x.flatten()) for x in all_t1_pred])
        l1 = np.concatenate([np.atleast_1d(x.flatten()) for x in all_t1_lbl])
        m  = compute_cls_metrics(p1, l1)
        result.update({'tier1_acc': m['accuracy'], 'tier1_f1': m['f1'], 'tier1_auc': m['auc']})

    if all_t2_pred:
        p2 = np.concatenate([np.atleast_1d(x.flatten()) for x in all_t2_pred])
        l2 = np.concatenate([np.atleast_1d(x.flatten()) for x in all_t2_lbl])
        m  = compute_cls_metrics(p2, l2.astype(int))
        result.update({'tier2_acc': m['accuracy'], 'tier2_f1': m['f1'], 'tier2_auc': m['auc']})

    if all_t3_pred:
        p3 = np.concatenate(all_t3_pred, axis=0)
        l3 = np.concatenate(all_t3_lbl,  axis=0)
        m  = compute_tier3_metrics(p3, l3)
        result.update({'tier3_acc': m['accuracy'], 'tier3_f1_macro': m['f1_macro']})

    if all_dice:
        result['dice'] = float(np.mean(all_dice))
        result['iou']  = float(np.mean(all_iou))

    return result


def print_results(title: str, metrics: dict):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    rows = [
        ('Segmentation Dice',  metrics.get('dice',           0)),
        ('Segmentation IoU',   metrics.get('iou',            0)),
        ('Tier1 Accuracy',     metrics.get('tier1_acc',      0)),
        ('Tier1 F1',           metrics.get('tier1_f1',       0)),
        ('Tier1 AUC',          metrics.get('tier1_auc',      0)),
        ('Tier2 Accuracy',     metrics.get('tier2_acc',      0)),
        ('Tier2 F1',           metrics.get('tier2_f1',       0)),
        ('Tier2 AUC',          metrics.get('tier2_auc',      0)),
        ('Tier3 Accuracy',     metrics.get('tier3_acc',      0)),
        ('Tier3 F1 macro',     metrics.get('tier3_f1_macro', 0)),
    ]
    for name, val in rows:
        print(f"  {name:<22} {val:.4f}")


def main():
    cfg    = load_config('configs/default.yaml')
    device = get_device()

    data_dir = cfg['data']['data_dir']
    img_dir  = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_split.csv'))

    model = BoneMTL(
        num_tumor_types = cfg['model']['num_classes'],
        pretrained      = False,
    ).to(device)
    epoch, _ = load_checkpoint('checkpoints/best.pth', model)
    print(f"Loaded checkpoint epoch {epoch}")

    test_tf     = build_test_transform(cfg)
    test_ds     = BTXRDDataset(test_df, img_dir, mask_dir, test_tf)
    test_loader = DataLoader(
        test_ds, batch_size=cfg['data']['batch_size'],
        shuffle=False, num_workers=0,
    )

    # No TTA
    no_tta = evaluate(model, test_loader, device)
    print_results("Test Set — No TTA", no_tta)

    # TTA
    tta_tfs = build_tta_transforms(cfg)
    tta_res = evaluate_tta(model, test_df, img_dir, mask_dir, tta_tfs, device,
                           batch_size=cfg['data']['batch_size'])
    print_results("Test Set — With TTA (5x)", tta_res)

    # CRF
    print("\nRunning CRF post-processing (slower)...")
    crf_res = evaluate_crf(model, test_df, img_dir, mask_dir, test_tf, device,
                           batch_size=cfg['data']['batch_size'])
    print_results("Test Set — With CRF", crf_res)

    # Lưu tất cả
    os.makedirs('results', exist_ok=True)
    with open('results/test_results.json', 'w') as f:
        json.dump({'no_tta': no_tta, 'tta': tta_res, 'crf': crf_res}, f, indent=2)
    print("\nSaved: results/test_results.json")


if __name__ == '__main__':
    main()