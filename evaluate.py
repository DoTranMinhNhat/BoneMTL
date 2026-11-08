import json
import os

import albumentations as A
import numpy as np
import torch
from PIL import Image

from src.inference import Collector, build_test_loader, load_model, predict
from src.metrics import compute_dice, compute_iou
from src.utils import get_device, load_config

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except ImportError:
    dcrf = None

RESULTS_PATH = 'results/test_results.json'

RESULT_ROWS = [
    ('Segmentation Dice', 'dice'),
    ('Segmentation IoU', 'iou'),
    ('Tier1 Accuracy', 'tier1_acc'),
    ('Tier1 F1', 'tier1_f1'),
    ('Tier1 AUC', 'tier1_auc'),
    ('Tier2 Accuracy', 'tier2_acc'),
    ('Tier2 F1', 'tier2_f1'),
    ('Tier2 AUC', 'tier2_auc'),
    ('Tier3 Accuracy', 'tier3_acc'),
    ('Tier3 F1 macro', 'tier3_f1_macro'),
]


@torch.no_grad()
def evaluate(model, loader, device, tta: bool = False) -> dict:
    model.eval()
    collector = Collector()

    for batch in loader:
        pred = predict(model, batch['image'].to(device), tta)
        collector.add_cls(pred, batch, device)

        hm = batch['has_mask'].to(device)
        if hm.sum() > 0:
            masks = batch['mask'].to(device)[hm]
            collector.add_seg(
                compute_dice(pred['mask'][hm], masks),
                compute_iou(pred['mask'][hm], masks),
            )

    return collector.result()


def apply_crf(image: np.ndarray, prob_map: np.ndarray, n_iter: int = 5,
              sxy_g: int = 3, sxy_b: int = 80, srgb: int = 13,
              compat: int = 3) -> np.ndarray:
    h, w = image.shape[:2]

    prob_fg = prob_map.astype(np.float32)
    probs = np.clip(np.stack([1.0 - prob_fg, prob_fg], axis=0),
                    1e-6, 1.0 - 1e-6)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary_from_softmax(probs))
    d.addPairwiseGaussian(sxy=sxy_g, compat=compat)
    d.addPairwiseBilateral(
        sxy=sxy_b, srgb=srgb, rgbim=np.ascontiguousarray(image), compat=compat
    )

    q = d.inference(n_iter)
    return np.argmax(q, axis=0).reshape(h, w).astype(np.uint8)


@torch.no_grad()
def evaluate_crf(model, loader, img_dir, size, device) -> dict:
    model.eval()
    collector = Collector()
    resize = A.Compose([A.Resize(size, size)])

    for batch in loader:
        pred = predict(model, batch['image'].to(device))
        collector.add_cls(pred, batch, device)

        logits = pred['mask'].cpu().numpy()
        gt_masks = batch['mask'].numpy()

        for j, img_id in enumerate(batch['image_id']):
            if not batch['has_mask'][j]:
                continue

            img = np.array(
                Image.open(os.path.join(img_dir, img_id)).convert('RGB')
            )
            img = np.ascontiguousarray(
                resize(image=img)['image'], dtype=np.uint8
            )

            prob_map = 1.0 / (1.0 + np.exp(-logits[j, 0]))
            refined = apply_crf(img, prob_map)

            gt = gt_masks[j, 0]
            inter = (refined * gt).sum()
            dice = float(2 * inter + 1e-6) / float(
                refined.sum() + gt.sum() + 1e-6
            )
            iou = float(inter + 1e-6) / float(
                refined.sum() + gt.sum() - inter + 1e-6
            )
            collector.add_seg(dice, iou)

    return collector.result()


def print_results(title: str, metrics: dict):
    print(f"\n{'=' * 50}\n  {title}\n{'=' * 50}")
    for name, key in RESULT_ROWS:
        print(f"  {name:<22} {metrics.get(key, 0):.4f}")


def record(results: dict, key: str, title: str, metrics: dict):
    results[key] = metrics
    print_results(title, metrics)
    os.makedirs('results', exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    cfg = load_config('configs/default.yaml')
    device = get_device()
    loader = build_test_loader(cfg)

    save_dir = cfg['checkpoint']['save_dir']
    results = {}

    model = load_model(os.path.join(save_dir, 'best.pth'), cfg, device)
    record(results, 'no_tta', 'Test set: best.pth, no TTA',
           evaluate(model, loader, device))
    record(results, 'tta', 'Test set: best.pth, TTA (5 views)',
           evaluate(model, loader, device, tta=True))

    if dcrf is None:
        print("\npydensecrf not installed, skipping CRF")
    else:
        print("\nRunning CRF post-processing (slower)...")
        img_dir = os.path.join(cfg['data']['data_dir'], 'images')
        record(results, 'crf', 'Test set: best.pth, CRF',
               evaluate_crf(model, loader, img_dir,
                            cfg['data']['img_size'], device))

    swa_path = os.path.join(save_dir, 'swa.pth')
    if os.path.exists(swa_path):
        swa_model = load_model(swa_path, cfg, device)
        record(results, 'swa', 'Test set: swa.pth',
               evaluate(swa_model, loader, device))
    else:
        print("\nswa.pth not found, skipping SWA evaluation")

    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == '__main__':
    main()