# src/trainer.py
import torch
import numpy as np
from tqdm import tqdm
from src.metrics import (
    compute_dice, compute_iou,
    compute_cls_metrics, compute_tier3_metrics,
    MetricTracker,
)
from src.utils import save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train 1 epoch, trả về dict metrics."""
    model.train()
    tracker = MetricTracker()
    pbar    = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        images  = batch['image'].to(device)
        outputs = model(images)
        losses  = criterion(outputs, batch, device)

        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = images.size(0)
        tracker.update('loss',    losses['total'].item(), bs)
        tracker.update('l_tier1', losses['l_tier1'],      bs)
        tracker.update('l_tier2', losses['l_tier2'],      bs)
        tracker.update('l_tier3', losses['l_tier3'],      bs)
        tracker.update('l_seg',   losses['l_seg'],        bs)

        with torch.no_grad():
            hm = batch['has_mask'].to(device)
            if hm.sum() > 0:
                tracker.update(
                    'dice',
                    compute_dice(
                        outputs['mask'][hm],
                        batch['mask'].to(device)[hm]
                    ),
                    int(hm.sum()),
                )

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.3f}",
            'dice': f"{tracker.result().get('dice', 0):.3f}",
        })

    return tracker.result()


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """Evaluate model trên val hoặc test loader."""
    model.eval()
    tracker = MetricTracker()
    all_t1_pred,  all_t1_lbl = [], []
    all_t2_pred,  all_t2_lbl = [], []
    all_t3_pred,  all_t3_lbl = [], []
    all_dice, all_iou         = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)

    for batch in pbar:
        images  = batch['image'].to(device)
        outputs = model(images)
        losses  = criterion(outputs, batch, device)

        bs = images.size(0)
        tracker.update('loss',    losses['total'].item(), bs)
        tracker.update('l_tier1', losses['l_tier1'],      bs)
        tracker.update('l_tier2', losses['l_tier2'],      bs)
        tracker.update('l_tier3', losses['l_tier3'],      bs)
        tracker.update('l_seg',   losses['l_seg'],        bs)

        # Tier 1 predictions
        all_t1_pred.append(
            torch.sigmoid(outputs['tier1']).cpu().numpy().squeeze()
        )
        all_t1_lbl.append(batch['tier1'].numpy().squeeze())

        # Tier 2 predictions
        m2 = (batch['tier2'] >= 0).squeeze(1)
        if m2.sum() > 0:
            all_t2_pred.append(
                torch.sigmoid(
                    outputs['tier2'][m2.to(device)]
                ).cpu().numpy().squeeze()
            )
            all_t2_lbl.append(batch['tier2'][m2].numpy().squeeze())

        # Tier 3 predictions
        m3 = batch['tier1'].squeeze(1).bool()
        if m3.sum() > 0:
            all_t3_pred.append(
                torch.softmax(
                    outputs['tier3'][m3.to(device)], dim=1
                ).cpu().numpy()
            )
            all_t3_lbl.append(batch['tier3'][m3].argmax(dim=1).numpy())

        # Segmentation metrics
        hm = batch['has_mask'].to(device)
        if hm.sum() > 0:
            all_dice.append(
                compute_dice(outputs['mask'][hm], batch['mask'].to(device)[hm])
            )
            all_iou.append(
                compute_iou(outputs['mask'][hm], batch['mask'].to(device)[hm])
            )

    result = tracker.result()

    if all_t1_pred:
        p1 = np.concatenate([np.atleast_1d(x) for x in all_t1_pred])
        l1 = np.concatenate([np.atleast_1d(x) for x in all_t1_lbl])
        m  = compute_cls_metrics(p1, l1)
        result.update({
            'tier1_acc': m['accuracy'],
            'tier1_f1':  m['f1'],
            'tier1_auc': m['auc'],
        })

    if all_t2_pred:
        p2 = np.concatenate([np.atleast_1d(x) for x in all_t2_pred])
        l2 = np.concatenate([np.atleast_1d(x) for x in all_t2_lbl])
        m  = compute_cls_metrics(p2, l2.astype(int))
        result.update({
            'tier2_acc': m['accuracy'],
            'tier2_f1':  m['f1'],
            'tier2_auc': m['auc'],
        })

    if all_t3_pred:
        p3 = np.concatenate(all_t3_pred, axis=0)
        l3 = np.concatenate(all_t3_lbl,  axis=0)
        m  = compute_tier3_metrics(p3, l3)
        result.update({
            'tier3_acc':      m['accuracy'],
            'tier3_f1_macro': m['f1_macro'],
        })

    if all_dice:
        result['dice'] = float(np.mean(all_dice))
        result['iou']  = float(np.mean(all_iou))

    return result


def train(model, train_loader, val_loader, criterion,
          optimizer, scheduler, device, cfg):
    """Training loop chính với early stopping và checkpoint."""
    epochs   = cfg['training']['epochs']
    save_dir = cfg['checkpoint']['save_dir']
    monitor  = cfg['checkpoint']['monitor']
    patience = 10

    best_score = -1.0
    no_improve = 0
    history    = {'train': [], 'val': []}

    print(f"Training {epochs} epochs | monitor: {monitor} | patience: {patience}")

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_m = validate(model, val_loader, criterion, device, epoch)
        scheduler.step(val_m.get('loss', 0))

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss: {train_m.get('loss', 0):.4f} | "
            f"val_loss: {val_m.get('loss', 0):.4f} | "
            f"dice: {val_m.get('dice', 0):.4f} | "
            f"t1_f1: {val_m.get('tier1_f1', 0):.4f} | "
            f"t2_f1: {val_m.get('tier2_f1', 0):.4f} | "
            f"t3_f1: {val_m.get('tier3_f1_macro', 0):.4f}"
        )

        history['train'].append(train_m)
        history['val'].append(val_m)

        monitor_key   = monitor.replace('val_', '')
        current_score = val_m.get(monitor_key, 0)

        if current_score > best_score:
            best_score = current_score
            no_improve = 0
            save_checkpoint(
                state={
                    'epoch':                epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics':              val_m,
                    'cfg':                  cfg,
                },
                save_dir=save_dir,
                filename='best.pth',
            )
            print(f"Checkpoint saved (epoch {epoch}, {monitor}: {best_score:.4f})")
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best {monitor}: {best_score:.4f}")
    return history