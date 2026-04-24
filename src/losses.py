# src/losses.py
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss = 1 - Dice coefficient.

    Dice = 2|A∩B| / (|A|+|B|) (foreground nhỏ hơn so với background nên phù hợp cho segmentation)
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = torch.sigmoid(pred).view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        inter  = (pred * target).sum(dim=1)
        dice   = (2.0 * inter + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    L_seg = α·Dice + (1-α)·BCE

    Dice (shape-level) + BCE (pixel-level).
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.dice  = DiceLoss()
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.dice(pred, target) + \
               (1 - self.alpha) * self.bce(pred, target)


class MultiTaskLoss(nn.Module):
    """
    Tổng loss đa nhiệm:
        L = λ1·L_tier1 + λ2·L_tier2 + λ3·L_tier3 + λ4·L_seg

    Tier1 : BCEWithLogitsLoss
    Tier2 : BCEWithLogitsLoss với pos_weight (malignant hiếm hơn benign)
    Tier3 : CrossEntropyLoss với class weights (9 class imbalanced)
    Seg   : Dice + BCE (chỉ tính trên ảnh có mask)

    Args:
        tier3_weights    : weight mỗi loại u, tỷ lệ nghịch tần suất
        tier2_pos_weight : weight cho malignant class
        lambda_*         : trọng số mỗi task
        device           : torch.device
    """

    def __init__(
        self,
        tier3_weights:    list,
        tier2_pos_weight: float = 4.5,
        lambda_tier1:     float = 1.0,
        lambda_tier2:     float = 1.0,
        lambda_tier3:     float = 1.0,
        lambda_seg:       float = 1.0,
        device:           torch.device = None,
    ):
        super().__init__()
        self.lambda_tier1 = lambda_tier1
        self.lambda_tier2 = lambda_tier2
        self.lambda_tier3 = lambda_tier3
        self.lambda_seg   = lambda_seg

        pw2 = torch.tensor([tier2_pos_weight])
        w3  = torch.tensor(tier3_weights, dtype=torch.float32)
        if device:
            pw2 = pw2.to(device)
            w3  = w3.to(device)

        self.loss_tier1 = nn.BCEWithLogitsLoss()
        self.loss_tier2 = nn.BCEWithLogitsLoss(pos_weight=pw2)
        self.loss_tier3 = nn.CrossEntropyLoss(weight=w3)
        self.loss_seg   = SegmentationLoss(alpha=0.5)

    def forward(self, outputs: dict, batch: dict, device: torch.device) -> dict:
        t1       = batch['tier1'].to(device)
        t2       = batch['tier2'].to(device)
        t3       = batch['tier3'].to(device)
        mask_gt  = batch['mask'].to(device)
        has_mask = batch['has_mask'].to(device)

        # Tier 1 loss
        l1 = self.loss_tier1(outputs['tier1'], t1)

        # Tier 2 loss
        m2 = (t2 >= 0).squeeze(1)
        l2 = self.loss_tier2(outputs['tier2'][m2], t2[m2]) \
             if m2.sum() > 0 else torch.tensor(0.0, device=device)

        # Tier 3 loss
        m3 = t1.squeeze(1).bool()
        l3 = self.loss_tier3(
            outputs['tier3'][m3], t3[m3].argmax(dim=1)
        ) if m3.sum() > 0 else torch.tensor(0.0, device=device)

        # Segmentation loss
        ls = self.loss_seg(outputs['mask'][has_mask], mask_gt[has_mask]) \
             if has_mask.sum() > 0 else torch.tensor(0.0, device=device)

        total = (self.lambda_tier1 * l1 + self.lambda_tier2 * l2 +
                 self.lambda_tier3 * l3 + self.lambda_seg   * ls)

        return {
            'total':   total,
            'l_tier1': l1.item(),
            'l_tier2': l2.item(),
            'l_tier3': l3.item(),
            'l_seg':   ls.item(),
        }