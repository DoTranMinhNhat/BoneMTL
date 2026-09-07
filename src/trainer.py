# src/trainer.py
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, update_bn
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

    result = tracker.result()

    if all_t1_pred:
        p1 = np.concatenate([np.atleast_1d(x) for x in all_t1_pred])
        l1 = np.concatenate([np.atleast_1d(x) for x in all_t1_lbl])
        m  = compute_cls_metrics(p1, l1)
        result.update({'tier1_acc': m['accuracy'], 'tier1_f1': m['f1'], 'tier1_auc': m['auc']})

    if all_t2_pred:
        p2 = np.concatenate([np.atleast_1d(x) for x in all_t2_pred])
        l2 = np.concatenate([np.atleast_1d(x) for x in all_t2_lbl])
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


def train(model, train_loader, val_loader, criterion,
          optimizer, scheduler, swa_model, swa_scheduler,
          swa_start, device, cfg, start_epoch=0):
    """
    Training loop với early stopping, Cosine Annealing, SWA và checkpoint.

    SWA (Stochastic Weight Averaging):
        Từ epoch swa_start trở đi, average weights model qua các epoch.
        Cuối training update BatchNorm stats cho SWA model.
        Thường cải thiện generalization 0.01-0.02 Dice miễn phí.
    """
    epochs   = cfg['training']['epochs']
    save_dir = cfg['checkpoint']['save_dir']
    monitor  = cfg['checkpoint']['monitor']
    patience = 10

    best_score  = -1.0
    no_improve  = 0
    swa_started = False
    history     = {'train': [], 'val': []}

    print(f"Training {epochs} epochs | monitor: {monitor} | "
          f"patience: {patience} | SWA from epoch: {swa_start}")

    for epoch in range(start_epoch + 1, epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # SWA — update averaged model và dùng SWA scheduler
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_started = True
            scheduler_mode = 'SWA'
        else:
            # Cosine Annealing trước khi SWA
            scheduler.step()
            scheduler_mode = f"lr={optimizer.param_groups[0]['lr']:.2e}"

        val_m = validate(model, val_loader, criterion, device, epoch)

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss: {train_m.get('loss', 0):.4f} | "
            f"val_loss: {val_m.get('loss', 0):.4f} | "
            f"dice: {val_m.get('dice', 0):.4f} | "
            f"t1_f1: {val_m.get('tier1_f1', 0):.4f} | "
            f"t2_f1: {val_m.get('tier2_f1', 0):.4f} | "
            f"t3_f1: {val_m.get('tier3_f1_macro', 0):.4f} | "
            f"{scheduler_mode}"
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
            print(f"  Checkpoint saved (epoch {epoch}, {monitor}: {best_score:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")

        save_checkpoint(
            state={
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics':              val_m,
                'cfg':                  cfg,
            },
            save_dir=save_dir,
            filename='last.pth',
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # SWA finalize — update BatchNorm statistics
    if swa_started:
        print("Updating SWA BatchNorm statistics...")
        update_bn(train_loader, swa_model, device=device)
        save_checkpoint(
            state={
                'epoch':             epochs,
                'model_state_dict':  swa_model.module.state_dict(),
                'metrics':           {},
                'cfg':               cfg,
            },
            save_dir=save_dir,
            filename='swa.pth',
        )
        print("Saved: checkpoints/swa.pth")

    print(f"Best {monitor}: {best_score:.4f}")
    return history