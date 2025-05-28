""" Utility functions for training, evaluation and visualization. """

from torch import nn
from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure as ssim

import torch


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute the SSIM (Structural Similarity Index) between predicted and clean images.
    Args:
        pred: predicted image
        target: ground truth image
    Returns:
        float: average SSIM score
    """

    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    return ssim(pred, target).item()

def total_loss_func(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute total loss = alpha * L1 + beta * (1 - SSIM)
    Args:
        pred: predicted image
        target: ground truth image
        alpha: weight for L1 loss
        beta: weight for 1 - SSIM
    Returns:
        total loss: torch scalar
    """
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    l1_loss = nn.functional.l1_loss(pred, target)
    ssim_loss = 1 - compute_ssim(pred, target)

    return alpha * l1_loss + beta * ssim_loss


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute the average PSNR (Peak Signal-to-Noise Ratio) between predicted and clean images.

    Args:
        pred: (B, C, H, W) predicted images, float tensor in [0, 1]
        target: (B, C, H, W) ground truth images, float tensor in [0, 1]

    Returns:
        Average PSNR over batch (float)
    """
    assert pred.shape == target.shape

    pred = pred.detach().cpu().clamp(0, 1).numpy()
    target = target.detach().cpu().clamp(0, 1).numpy()

    pred = pred.transpose(0, 2, 3, 1)   # (B, H, W, C)
    target = target.transpose(0, 2, 3, 1)

    total_psnr = 0.0
    for i in range(pred.shape[0]):
        total_psnr += peak_signal_noise_ratio(target[i], pred[i], data_range=1.0)

    return total_psnr / pred.shape[0]


def tqdm_bar(mode: str, pbar, target: float = 0.0, cur_epoch: int = 0, epochs: int = 0) -> None:
    """
    Update the tqdm progress bar with custom format.

    Args:
        mode (str): Current mode ('Train', 'Val', 'Test').
        pbar: tqdm progress bar instance.
        target (float): Current loss value.
        cur_epoch (int): Current epoch.
        epochs (int): Total number of epochs.
    """
    if mode == 'Test' or mode == 'Evaluation':
        pbar.set_description(f"({mode})", refresh=False)
    else:
        pbar.set_description(f"({mode}) Epoch {cur_epoch}/{epochs}", refresh=False)
        pbar.set_postfix(loss=float(target), refresh=False)
    pbar.refresh()

def print_model_params(model: nn.Module) -> None:
    """
    Print the model architecture and total number of parameters.

    Args:
        model (nn.Module): PyTorch model
    """
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print("# Parameters:", total_params)
