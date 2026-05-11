from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

from .constants import FIELD_STRENGTH, FIELDS, MOD_TO_INDEX, MODALITIES


def _as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


class FieldModalityConditioner(nn.Module):
    """Continuous field-strength + modality encoder.

    Field values are encoded by Fourier features of log(Tesla). This lets the
    same model cover all field pairs without hard-coding pair-specific heads.
    """

    def __init__(self, cond_dim: int = 128, fourier_bands: int = 6):
        super().__init__()
        self.cond_dim = cond_dim
        self.fourier_bands = fourier_bands
        in_dim = 2 * (1 + 2 * fourier_bands) + len(MODALITIES)
        self.net = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
        )

    def _field_features(self, fields: Sequence[str], device: torch.device) -> torch.Tensor:
        vals = []
        for f in fields:
            if f not in FIELD_STRENGTH:
                raise KeyError(f"Unknown field '{f}'. Expected one of {FIELDS}")
            vals.append(math.log(float(FIELD_STRENGTH[f]) + 1e-8))
        x = torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(1)
        feats = [x]
        for k in range(self.fourier_bands):
            freq = 2.0**k
            feats.append(torch.sin(freq * x))
            feats.append(torch.cos(freq * x))
        return torch.cat(feats, dim=1)

    def _mod_features(self, modalities: Sequence[str], device: torch.device) -> torch.Tensor:
        out = torch.zeros((len(modalities), len(MODALITIES)), dtype=torch.float32, device=device)
        for i, m in enumerate(modalities):
            if m not in MOD_TO_INDEX:
                raise KeyError(f"Unknown modality '{m}'. Expected one of {MODALITIES}")
            out[i, MOD_TO_INDEX[m]] = 1.0
        return out

    def forward(self, source_fields, target_fields, modalities, device=None) -> torch.Tensor:
        source_fields = _as_list(source_fields)
        target_fields = _as_list(target_fields)
        modalities = _as_list(modalities)
        n = max(len(source_fields), len(target_fields), len(modalities))
        if len(source_fields) == 1:
            source_fields = source_fields * n
        if len(target_fields) == 1:
            target_fields = target_fields * n
        if len(modalities) == 1:
            modalities = modalities * n
        if not (len(source_fields) == len(target_fields) == len(modalities)):
            raise ValueError("source_fields, target_fields and modalities must broadcast to the same length")
        if device is None:
            device = next(self.parameters()).device
        src = self._field_features(source_fields, device)
        tgt = self._field_features(target_fields, device)
        mod = self._mod_features(modalities, device)
        return self.net(torch.cat([src, tgt, mod], dim=1))


class FiLM3D(nn.Module):
    def __init__(self, num_channels: int, cond_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * num_channels)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        shape = (x.shape[0], x.shape[1], 1, 1, 1)
        gamma = gamma.view(shape)
        beta = beta.view(shape)
        return x * (1.0 + gamma) + beta
