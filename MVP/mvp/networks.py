from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import FiLM3D, FieldModalityConditioner


def _group_count(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def make_coord_channels(
    shape: Sequence[int],
    starts: Optional[Sequence[int]] = None,
    full_shape: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create 3 coordinate channels with values in [-1, 1].

    Args:
        shape: patch shape (D, H, W).
        starts: start index of the patch in full volume, default all zero.
        full_shape: full volume shape. If omitted, coordinates are local patch coordinates.
    Returns:
        Tensor of shape (3, D, H, W).
    """
    d, h, w = [int(v) for v in shape]
    starts = starts or (0, 0, 0)
    full_shape = full_shape or shape
    axes = []
    for n, start, full in zip((d, h, w), starts, full_shape):
        idx = torch.arange(start, start + n, dtype=torch.float32, device=device)
        denom = max(float(full - 1), 1.0)
        axes.append(idx / denom * 2.0 - 1.0)
    zz, yy, xx = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return torch.stack([zz, yy, xx], dim=0)


class ConvFiLMBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.film1 = FiLM3D(out_ch, cond_dim)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.film2 = FiLM3D(out_ch, cond_dim)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.film1(x, cond)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film2(x, cond)
        x = self.act(x)
        return x


class ConditionalUNet3D(nn.Module):
    """Field-conditioned 3D U-Net for paired MRI translation.

    Use a small z-depth patch, e.g. ``patch_size: [16, 192, 192]``, to run the
    same network as a 2.5D model when full 3D patches are too expensive.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 24,
        levels: int = 4,
        cond_dim: int = 128,
        use_coords: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_coords = use_coords
        self.conditioner = FieldModalityConditioner(cond_dim=cond_dim)
        model_in = in_ch + (3 if use_coords else 0)
        chans = [base_ch * (2**i) for i in range(levels)]
        self.enc = nn.ModuleList()
        prev = model_in
        for ch in chans:
            self.enc.append(ConvFiLMBlock3D(prev, ch, cond_dim, dropout=dropout))
            prev = ch
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvFiLMBlock3D(chans[-1], chans[-1] * 2, cond_dim, dropout=dropout)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = chans[-1] * 2
        for ch in reversed(chans):
            self.up.append(nn.ConvTranspose3d(prev, ch, 2, stride=2))
            self.dec.append(ConvFiLMBlock3D(ch * 2, ch, cond_dim, dropout=dropout))
            prev = ch
        self.out = nn.Conv3d(chans[0], out_ch, 1)

    def _condition(self, source_fields, target_fields, modalities, device) -> torch.Tensor:
        return self.conditioner(source_fields, target_fields, modalities, device=device)

    def forward(
        self,
        x: torch.Tensor,
        source_fields,
        target_fields,
        modalities,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_coords:
            if coords is None:
                coords = make_coord_channels(x.shape[-3:], device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
            x = torch.cat([x, coords.to(dtype=x.dtype, device=x.device)], dim=1)
        cond = self._condition(source_fields, target_fields, modalities, x.device)
        skips = []
        h = x
        for i, block in enumerate(self.enc):
            h = block(h, cond)
            skips.append(h)
            if i != len(self.enc) - 1:
                h = self.pool(h)
        h = self.pool(h)
        h = self.bottleneck(h, cond)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            # Pad/crop for odd shapes after pooling.
            if h.shape[-3:] != skip.shape[-3:]:
                dz = skip.shape[-3] - h.shape[-3]
                dy = skip.shape[-2] - h.shape[-2]
                dx = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, [0, max(dx, 0), 0, max(dy, 0), 0, max(dz, 0)])
                h = h[..., : skip.shape[-3], : skip.shape[-2], : skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = dec(h, cond)
        return torch.sigmoid(self.out(h))


def _gaussian_kernel3d(sigma: float, device=None, dtype=None) -> torch.Tensor:
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    x = torch.arange(-radius, radius + 1, dtype=dtype or torch.float32, device=device)
    g = torch.exp(-(x**2) / (2 * sigma * sigma))
    g = g / g.sum()
    k = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return k[None, None]


class GaussianBlurBank3D(nn.Module):
    """Fixed Gaussian PSF bank with condition-predicted mixture weights."""

    def __init__(self, sigmas: Sequence[float] = (0.35, 0.7, 1.1, 1.6, 2.3)):
        super().__init__()
        self.sigmas = tuple(float(s) for s in sigmas)
        for i, sigma in enumerate(self.sigmas):
            k = _gaussian_kernel3d(sigma)
            self.register_buffer(f"kernel_{i}", k, persistent=False)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for i, _ in enumerate(self.sigmas):
            k = getattr(self, f"kernel_{i}").to(device=x.device, dtype=x.dtype)
            pad = k.shape[-1] // 2
            blurred = F.conv3d(F.pad(x, [pad, pad, pad, pad, pad, pad], mode="replicate"), k)
            out = out + blurred * weights[:, i].view(-1, 1, 1, 1, 1)
        return out


class FieldDegradationSimulator(nn.Module):
    """Physics-inspired learnable field degradation model.

    It intentionally has limited capacity: PSF blur bank + monotonic-ish
    intensity calibration + small residual CNN. This makes it useful for
    generating anatomy-preserving pseudo pairs from retrospective target images.
    """

    def __init__(
        self,
        cond_dim: int = 128,
        residual_ch: int = 16,
        sigmas: Sequence[float] = (0.35, 0.7, 1.1, 1.6, 2.3),
        use_coords: bool = True,
    ):
        super().__init__()
        self.use_coords = use_coords
        self.conditioner = FieldModalityConditioner(cond_dim=cond_dim)
        self.blur = GaussianBlurBank3D(sigmas=sigmas)
        self.param_head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, len(sigmas) + 4),
        )
        res_in = 1 + (3 if use_coords else 0)
        self.residual = nn.Sequential(
            nn.Conv3d(res_in, residual_ch, 3, padding=1),
            nn.GroupNorm(_group_count(residual_ch), residual_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(residual_ch, residual_ch, 3, padding=1),
            nn.GroupNorm(_group_count(residual_ch), residual_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(residual_ch, 1, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        from_fields,
        to_fields,
        modalities,
        coords: Optional[torch.Tensor] = None,
        add_noise: bool = False,
        noise_scale: float = 0.0,
    ) -> torch.Tensor:
        cond = self.conditioner(from_fields, to_fields, modalities, device=x.device)
        params = self.param_head(cond)
        weights = torch.softmax(params[:, : len(self.blur.sigmas)], dim=1)
        scale = 0.5 + F.softplus(params[:, len(self.blur.sigmas) + 0]).view(-1, 1, 1, 1, 1)
        bias = 0.15 * torch.tanh(params[:, len(self.blur.sigmas) + 1]).view(-1, 1, 1, 1, 1)
        quad = 0.25 * torch.tanh(params[:, len(self.blur.sigmas) + 2]).view(-1, 1, 1, 1, 1)
        res_gain = 0.15 * torch.sigmoid(params[:, len(self.blur.sigmas) + 3]).view(-1, 1, 1, 1, 1)

        y = self.blur(x, weights)
        y = scale * y + quad * y * (1.0 - y) + bias
        if self.use_coords:
            if coords is None:
                coords = make_coord_channels(x.shape[-3:], device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
            res_in = torch.cat([y, coords.to(device=x.device, dtype=x.dtype)], dim=1)
        else:
            res_in = y
        y = y + res_gain * self.residual(res_in)
        if add_noise and noise_scale > 0:
            # Synthetic generation only. Training should normally keep this off.
            sigma = float(noise_scale)
            y = y + sigma * torch.randn_like(y) * torch.sqrt(torch.clamp(y, min=1e-4))
        return torch.clamp(y, 0.0, 1.0)


def build_model_from_config(cfg: dict) -> ConditionalUNet3D:
    mc = cfg.get("model", {})
    return ConditionalUNet3D(
        in_ch=mc.get("in_ch", 1),
        out_ch=mc.get("out_ch", 1),
        base_ch=mc.get("base_ch", 24),
        levels=mc.get("levels", 4),
        cond_dim=mc.get("cond_dim", 128),
        use_coords=mc.get("use_coords", True),
        dropout=mc.get("dropout", 0.0),
    )


def build_fds_from_config(cfg: dict) -> FieldDegradationSimulator:
    fc = cfg.get("fds", {})
    return FieldDegradationSimulator(
        cond_dim=fc.get("cond_dim", 128),
        residual_ch=fc.get("residual_ch", 16),
        sigmas=fc.get("sigmas", (0.35, 0.7, 1.1, 1.6, 2.3)),
        use_coords=fc.get("use_coords", True),
    )
