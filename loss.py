import torch
import torch.nn.functional as F


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