import torch
import torch.nn.functional as F
import torch.nn as nn


def info_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Compute InfoNCE loss between query and key.
    Args:
        query: [N, C]
        key:   [N, C] (positives are aligned by index)
        temperature: scaling factor
    Returns:
        scalar loss
    """
    assert query.dim() == 2 and key.dim() == 2, "query/key must be 2D"
    assert query.size(0) == key.size(0) and query.size(1) == key.size(1), "shape mismatch"
    q = F.normalize(query, dim=1)
    k = F.normalize(key, dim=1)
    logits = (q @ k.t()) / max(temperature, 1e-6)
    targets = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, targets)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for multi-class segmentation.
    Args:
        logits: [B, C, H, W]
        targets: [B, H, W] with class indices, may include ignore_index
        ignore_index: label to ignore
        eps: numerical stability
    Returns:
        scalar dice loss (1 - mean_dice)
    """
    if logits.dim() != 4:
        raise ValueError("logits must be [B,C,H,W]")
    B, C, H, W = logits.shape
    probs = F.softmax(logits, dim=1)
    valid_mask = (targets != ignore_index).unsqueeze(1)  # [B,1,H,W]
    safe_targets = targets.clone()
    safe_targets[safe_targets == ignore_index] = 0
    tgt_onehot = torch.zeros((B, C, H, W), dtype=probs.dtype, device=probs.device)
    tgt_onehot.scatter_(1, safe_targets.unsqueeze(1), 1)
    probs = probs * valid_mask
    tgt_onehot = tgt_onehot * valid_mask
    dims = (0, 2, 3)
    intersection = (probs * tgt_onehot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + tgt_onehot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    dice_mean = dice.mean()
    return 1.0 - dice_mean


class CombinedSegLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, ignore_index: int = -1, class_weights=None):
        super().__init__()
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        weight_tensor = None
        if self.class_weights is not None:
            weight_tensor = torch.tensor(self.class_weights, dtype=torch.float32, device=logits.device)
        ce = F.cross_entropy(logits, targets, weight=weight_tensor, ignore_index=self.ignore_index)
        dl = dice_loss(logits, targets, ignore_index=self.ignore_index)
        total = self.ce_w * ce + self.dice_w * dl
        return total, {"ce": ce.detach(), "dice": dl.detach()}