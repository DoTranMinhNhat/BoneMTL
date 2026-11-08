import numpy as np
import torch
from tqdm import tqdm

from src.metrics import (
    MetricTracker, compute_cls_metrics, compute_dice, compute_iou,
    compute_tier3_metrics,
)
from src.utils import save_checkpoint

LOSS_KEYS = ('l_tier1', 'l_tier2', 'l_tier3', 'l_seg')


def mixup_batch(batch: dict, alpha: float = 0.2,
                device: torch.device = None) -> dict:
    images = batch['image'].to(device)
    tumor_idx = torch.where(batch['tier1'].squeeze(1) == 1)[0]

    if len(tumor_idx) < 2:
        return batch

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    perm = tumor_idx[torch.randperm(len(tumor_idx))]

    images[tumor_idx] = lam * images[tumor_idx] + (1 - lam) * images[perm]
    batch['image'] = images

    t3 = batch['tier3'].clone()
    t3[tumor_idx] = lam * t3[tumor_idx] + (1 - lam) * t3[perm]
    batch['tier3'] = t3

    masks = batch['mask'].to(device)
    has_mask = batch['has_mask']
    both = has_mask[tumor_idx] & has_mask[perm]
    if both.sum() > 0:
        idx, idx_perm = tumor_idx[both], perm[both]
        masks[idx] = lam * masks[idx] + (1 - lam) * masks[idx_perm]
    batch['mask'] = masks

    return batch


def _track_losses(tracker: MetricTracker, losses: dict, n: int):
    tracker.update('loss', losses['total'].item(), n)
    for key in LOSS_KEYS:
        tracker.update(key, losses[key], n)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    use_mixup: bool = False, mixup_alpha: float = 0.2):
    model.train()
    tracker = MetricTracker()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        if use_mixup:
            batch = mixup_batch(batch, alpha=mixup_alpha, device=device)

        images = batch['image'].to(device)
        outputs = model(images)
        losses = criterion(outputs, batch, device)

        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _track_losses(tracker, losses, images.size(0))

        with torch.no_grad():
            hm = batch['has_mask'].to(device)
            if hm.sum() > 0:
                masks = batch['mask'].to(device)
                tracker.update(
                    'dice',
                    compute_dice(outputs['mask'][hm], masks[hm]),
                    int(hm.sum()),
                )

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.3f}",
            'dice': f"{tracker.result().get('dice', 0):.3f}",
        })

    return tracker.result()


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    tracker = MetricTracker()
    t1_pred, t1_lbl = [], []
    t2_pred, t2_lbl = [], []
    t3_pred, t3_lbl = [], []
    dices, ious = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)
        outputs = model(images)
        losses = criterion(outputs, batch, device)

        _track_losses(tracker, losses, images.size(0))

        t1_pred.append(torch.sigmoid(outputs['tier1']).cpu().numpy().squeeze())
        t1_lbl.append(batch['tier1'].numpy().squeeze())

        m2 = (batch['tier2'] >= 0).squeeze(1)
        if m2.sum() > 0:
            logits = outputs['tier2'][m2.to(device)]
            t2_pred.append(torch.sigmoid(logits).cpu().numpy().squeeze())
            t2_lbl.append(batch['tier2'][m2].numpy().squeeze())

        m3 = batch['tier1'].squeeze(1).bool()
        if m3.sum() > 0:
            logits = outputs['tier3'][m3.to(device)]
            t3_pred.append(torch.softmax(logits, dim=1).cpu().numpy())
            t3_lbl.append(batch['tier3'][m3].argmax(dim=1).numpy())

        hm = batch['has_mask'].to(device)
        if hm.sum() > 0:
            masks = batch['mask'].to(device)
            dices.append(compute_dice(outputs['mask'][hm], masks[hm]))
            ious.append(compute_iou(outputs['mask'][hm], masks[hm]))

    result = tracker.result()

    def flatten(parts):
        return np.concatenate([np.atleast_1d(x) for x in parts])

    if t1_pred:
        m = compute_cls_metrics(flatten(t1_pred), flatten(t1_lbl))
        result.update({
            'tier1_acc': m['accuracy'],
            'tier1_f1': m['f1'],
            'tier1_auc': m['auc'],
        })

    if t2_pred:
        m = compute_cls_metrics(
            flatten(t2_pred), flatten(t2_lbl).astype(int)
        )
        result.update({
            'tier2_acc': m['accuracy'],
            'tier2_f1': m['f1'],
            'tier2_auc': m['auc'],
        })

    if t3_pred:
        m = compute_tier3_metrics(
            np.concatenate(t3_pred, axis=0), np.concatenate(t3_lbl, axis=0)
        )
        result.update({
            'tier3_acc': m['accuracy'],
            'tier3_f1_macro': m['f1_macro'],
        })

    if dices:
        result['dice'] = float(np.mean(dices))
        result['iou'] = float(np.mean(ious))

    return result


def train(model, train_loader, val_loader, criterion,
          optimizer, scheduler, swa_model, swa_scheduler,
          swa_start, device, cfg, start_epoch=0):
    epochs = cfg['training']['epochs']
    save_dir = cfg['checkpoint']['save_dir']
    monitor = cfg['checkpoint']['monitor']
    patience = 10
    use_mixup = cfg['training'].get('use_mixup', True)
    mixup_alpha = cfg['training'].get('mixup_alpha', 0.2)

    best_score = -1.0
    no_improve = 0
    swa_started = False
    history = {'train': [], 'val': []}

    print(f"Training {epochs} epochs | monitor: {monitor} | "
          f"patience: {patience} | SWA from epoch: {swa_start} | "
          f"Mixup: {use_mixup} (alpha={mixup_alpha})")

    for epoch in range(start_epoch + 1, epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha,
        )
        val_m = validate(model, val_loader, criterion, device, epoch)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_started = True
            lr_info = 'SWA'
        else:
            scheduler.step(val_m.get('loss', 0))
            lr_info = f"lr={optimizer.param_groups[0]['lr']:.2e}"

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss: {train_m.get('loss', 0):.4f} | "
            f"val_loss: {val_m.get('loss', 0):.4f} | "
            f"dice: {val_m.get('dice', 0):.4f} | "
            f"t1_f1: {val_m.get('tier1_f1', 0):.4f} | "
            f"t2_f1: {val_m.get('tier2_f1', 0):.4f} | "
            f"t3_f1: {val_m.get('tier3_f1_macro', 0):.4f} | "
            f"{lr_info}"
        )

        history['train'].append(train_m)
        history['val'].append(val_m)

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_m,
            'cfg': cfg,
        }

        current_score = val_m.get(monitor.replace('val_', ''), 0)
        if current_score > best_score:
            best_score = current_score
            no_improve = 0
            save_checkpoint(state=state, save_dir=save_dir,
                            filename='best.pth')
            print(f"  Checkpoint saved (epoch {epoch}, "
                  f"{monitor}: {best_score:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")

        save_checkpoint(state=state, save_dir=save_dir, filename='last.pth')

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if swa_started:
        print("Updating SWA BatchNorm statistics...")
        swa_model.train()
        with torch.no_grad():
            for batch in train_loader:
                swa_model(batch['image'].to(device))
        save_checkpoint(
            state={
                'epoch': epochs,
                'model_state_dict': swa_model.module.state_dict(),
                'metrics': {},
                'cfg': cfg,
            },
            save_dir=save_dir,
            filename='swa.pth',
        )
        print(f"Saved: {save_dir}/swa.pth")
    else:
        print("SWA never started, swa.pth not saved")

    print(f"Best {monitor}: {best_score:.4f}")
    return history