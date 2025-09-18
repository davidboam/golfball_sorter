import torch
import torch.nn.functional as F


def quadratic_weighted_kappa(y_true: torch.Tensor, y_pred: torch.Tensor, n_grades: int) -> float:
    # Stub: returns 0.0; implement later
    return 0.0


def auroc_stub(scores: torch.Tensor, labels: torch.Tensor) -> float:
    # Stub
    return 0.0


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # logits: [B,C,H,W], target: [B,C,H,W] in {0,1}
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=(0, 2, 3))
    den = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + eps
    dice = 1 - (num + eps) / (den + eps)
    return dice.mean()


def ece_stub(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return 0.0

