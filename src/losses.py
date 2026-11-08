import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (pred * target).sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.alpha * self.dice(pred, target)
                + (1 - self.alpha) * self.bce(pred, target))


class LabelSmoothingBCE(nn.Module):
    def __init__(self, epsilon: float = 0.1, pos_weight: torch.Tensor = None):
        super().__init__()
        self.epsilon = epsilon
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smooth = targets * (1 - self.epsilon) + self.epsilon * (1 - targets)
        return F.binary_cross_entropy_with_logits(
            inputs, smooth, pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.weight is not None:
            focal = self.weight[targets] * focal
        return focal.mean()


class MultiTaskLoss(nn.Module):
    """
    L = l1*L_tier1 + l2*L_tier2 + l3*L_tier3 + l4*L_seg + l_aux*(L_aux3 + L_aux2)
    Tier 2 is computed on tumor samples only (target >= 0)
    Tier 3 on tumor samples only, and segmentation terms on samples with a mask.
    """

    def __init__(
        self,
        tier3_weights: list,
        tier2_pos_weight: float = 4.5,
        lambda_tier1: float = 1.0,
        lambda_tier2: float = 1.0,
        lambda_tier3: float = 1.0,
        lambda_seg: float = 2.0,
        focal_gamma: float = 2.0,
        lambda_aux: float = 0.4,
        label_smoothing: float = 0.1,
        device: torch.device = None,
    ):
        super().__init__()
        self.lambda_tier1 = lambda_tier1
        self.lambda_tier2 = lambda_tier2
        self.lambda_tier3 = lambda_tier3
        self.lambda_seg = lambda_seg
        self.lambda_aux = lambda_aux

        pw2 = torch.tensor([tier2_pos_weight])
        w3 = torch.tensor(tier3_weights, dtype=torch.float32)
        if device:
            pw2 = pw2.to(device)
            w3 = w3.to(device)

        self.loss_tier1 = LabelSmoothingBCE(epsilon=label_smoothing)
        self.loss_tier2 = LabelSmoothingBCE(
            epsilon=label_smoothing, pos_weight=pw2
        )
        self.loss_tier3 = FocalLoss(weight=w3, gamma=focal_gamma)
        self.loss_seg = SegmentationLoss(alpha=0.5)

    def forward(self, outputs: dict, batch: dict, device: torch.device) -> dict:
        t1 = batch['tier1'].to(device)
        t2 = batch['tier2'].to(device)
        t3 = batch['tier3'].to(device)
        mask_gt = batch['mask'].to(device)
        has_mask = batch['has_mask'].to(device)

        zero = torch.tensor(0.0, device=device)

        l1 = self.loss_tier1(outputs['tier1'], t1)

        m2 = (t2 >= 0).squeeze(1)
        l2 = (self.loss_tier2(outputs['tier2'][m2], t2[m2])
              if m2.sum() > 0 else zero)

        m3 = t1.squeeze(1).bool()
        l3 = (self.loss_tier3(outputs['tier3'][m3], t3[m3].argmax(dim=1))
              if m3.sum() > 0 else zero)

        if has_mask.sum() > 0:
            gt = mask_gt[has_mask]
            ls = self.loss_seg(outputs['mask'][has_mask], gt)
            l_aux3 = self.loss_seg(outputs['aux3'][has_mask], gt)
            l_aux2 = self.loss_seg(outputs['aux2'][has_mask], gt)
        else:
            ls = l_aux3 = l_aux2 = zero

        l_aux = self.lambda_aux * (l_aux3 + l_aux2)

        total = (self.lambda_tier1 * l1 + self.lambda_tier2 * l2
                 + self.lambda_tier3 * l3 + self.lambda_seg * ls + l_aux)

        return {
            'total': total,
            'l_tier1': l1.item(),
            'l_tier2': l2.item(),
            'l_tier3': l3.item(),
            'l_seg': ls.item(),
            'l_aux': l_aux.item(),
        }