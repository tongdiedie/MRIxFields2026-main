from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def nrmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = (pred - target).reshape(pred.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1)
    return (torch.linalg.norm(diff, dim=1) / (torch.linalg.norm(tgt, dim=1) + eps)).mean()


def _ssim2d(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable SSIM for Bx1xHxW tensors in [0,1]."""
    pad = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return ssim_map.mean()


def ssim_slice_loss(pred: torch.Tensor, target: torch.Tensor, axis: int = 2) -> torch.Tensor:
    """1 - SSIM on the central 2D slice of a 3D patch.

    Tensor shape is B,C,D,H,W. ``axis`` follows NumPy volume axes: 0=D,
    1=H, 2=W; in tensor dimensions this maps to 2,3,4.
    """
    if pred.ndim == 4:
        return 1.0 - _ssim2d(pred, target)
    if pred.ndim != 5:
        raise ValueError(f"Expected 4D/5D tensor, got {pred.shape}")
    dim = {0: 2, 1: 3, 2: 4}[axis]
    idx = pred.shape[dim] // 2
    if dim == 2:
        p, t = pred[:, :, idx, :, :], target[:, :, idx, :, :]
    elif dim == 3:
        p, t = pred[:, :, :, idx, :], target[:, :, :, idx, :]
    else:
        p, t = pred[:, :, :, :, idx], target[:, :, :, :, idx]
    return 1.0 - _ssim2d(p, t)


class LPIPSSliceLoss(nn.Module):
    """LPIPS on one central slice per 3D patch.

    This mirrors the official evaluator's 2D-slice LPIPS idea without paying the
    cost of all slices during training.
    """

    def __init__(self, net: str = "alex", axis: int = 2, device: Optional[torch.device] = None):
        super().__init__()
        try:
            import lpips
        except Exception as e:  # pragma: no cover - depends on environment
            raise RuntimeError("LPIPS loss requested but package 'lpips' is not installed") from e
        self.axis = axis
        self.fn = lpips.LPIPS(net=net)
        self.fn.eval()
        for p in self.fn.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.fn.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 5:
            dim = {0: 2, 1: 3, 2: 4}[self.axis]
            idx = pred.shape[dim] // 2
            if dim == 2:
                pred, target = pred[:, :, idx, :, :], target[:, :, idx, :, :]
            elif dim == 3:
                pred, target = pred[:, :, :, idx, :], target[:, :, :, idx, :]
            else:
                pred, target = pred[:, :, :, :, idx], target[:, :, :, :, idx]
        pred3 = pred.repeat(1, 3, 1, 1) * 2.0 - 1.0
        tgt3 = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.fn(pred3, tgt3).mean()


class MVPReconstructionLoss(nn.Module):
    def __init__(self, weights: Dict[str, float], lpips_axis: int = 2, lpips_net: str = "alex"):
        super().__init__()
        self.weights = {k: float(v) for k, v in weights.items() if float(v) != 0.0}
        self.lpips_axis = lpips_axis
        self.lpips_net = lpips_net
        self.lpips_loss: Optional[LPIPSSliceLoss] = None

    def _maybe_init_lpips(self, device: torch.device):
        if self.weights.get("lpips", 0.0) > 0 and self.lpips_loss is None:
            self.lpips_loss = LPIPSSliceLoss(net=self.lpips_net, axis=self.lpips_axis, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        self._maybe_init_lpips(pred.device)
        total = pred.new_tensor(0.0)
        logs: Dict[str, float] = {}
        if self.weights.get("l1", 0.0) > 0:
            val = l1_loss(pred, target)
            total = total + self.weights["l1"] * val
            logs["l1"] = float(val.detach().cpu())
        if self.weights.get("nrmse", 0.0) > 0:
            val = nrmse_loss(pred, target)
            total = total + self.weights["nrmse"] * val
            logs["nrmse"] = float(val.detach().cpu())
        if self.weights.get("ssim", 0.0) > 0:
            val = ssim_slice_loss(pred, target, axis=self.lpips_axis)
            total = total + self.weights["ssim"] * val
            logs["ssim_loss"] = float(val.detach().cpu())
        if self.weights.get("lpips", 0.0) > 0:
            assert self.lpips_loss is not None
            val = self.lpips_loss(pred, target)
            total = total + self.weights["lpips"] * val
            logs["lpips"] = float(val.detach().cpu())
        logs["total"] = float(total.detach().cpu())
        return total, logs
