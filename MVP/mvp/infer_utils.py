from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .networks import make_coord_channels


def _starts_1d(n: int, patch: int, overlap: float) -> List[int]:
    if n <= patch:
        return [0]
    stride = max(int(round(patch * (1.0 - overlap))), 1)
    starts = list(range(0, n - patch + 1, stride))
    if starts[-1] != n - patch:
        starts.append(n - patch)
    return starts


def _pad_to_patch(vol: np.ndarray, patch_size: Sequence[int]) -> tuple[np.ndarray, tuple[int, int, int]]:
    pads = []
    for n, p in zip(vol.shape, patch_size):
        extra = max(int(p) - int(n), 0)
        pads.append((0, extra))
    if any(a for _, a in pads):
        vol = np.pad(vol, pads, mode="constant", constant_values=0)
    return vol, tuple(int(n) for n, _ in pads)


def _importance_map(patch_size: Sequence[int]) -> np.ndarray:
    # Separable Hann window; fallback to ones on tiny axes.
    maps = []
    for p in patch_size:
        if p <= 2:
            maps.append(np.ones((p,), dtype=np.float32))
        else:
            w = np.hanning(p).astype(np.float32)
            w = np.maximum(w, 1e-3)
            maps.append(w)
    imp = maps[0][:, None, None] * maps[1][None, :, None] * maps[2][None, None, :]
    return imp.astype(np.float32)


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    volume: np.ndarray,
    source_field: str,
    target_field: str,
    modality: str,
    patch_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    device: torch.device | str = "cuda",
    amp: bool = True,
    use_coords: bool = True,
) -> np.ndarray:
    """Run field-conditioned 3D sliding-window inference on one volume."""
    device = torch.device(device)
    model.eval()
    patch_size = tuple(int(v) for v in patch_size)
    original_shape = tuple(int(v) for v in volume.shape)
    vol, _ = _pad_to_patch(volume.astype(np.float32, copy=False), patch_size)
    full_shape = vol.shape
    starts = [
        _starts_1d(full_shape[0], patch_size[0], overlap),
        _starts_1d(full_shape[1], patch_size[1], overlap),
        _starts_1d(full_shape[2], patch_size[2], overlap),
    ]
    out = np.zeros(full_shape, dtype=np.float32)
    weight = np.zeros(full_shape, dtype=np.float32)
    imp = _importance_map(patch_size)

    patch_batch = []
    coord_batch = []
    start_batch = []

    def flush():
        nonlocal patch_batch, coord_batch, start_batch, out, weight
        if not patch_batch:
            return
        x = torch.from_numpy(np.stack(patch_batch)[:, None]).to(device=device, dtype=torch.float32)
        coords = None
        if use_coords:
            coords = torch.stack(coord_batch).to(device=device, dtype=torch.float32)
        src_fields = [source_field] * x.shape[0]
        tgt_fields = [target_field] * x.shape[0]
        mods = [modality] * x.shape[0]
        enabled_amp = bool(amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=enabled_amp):
            pred = model(x, src_fields, tgt_fields, mods, coords=coords)
        pred_np = pred.squeeze(1).detach().float().cpu().numpy()
        for p, st in zip(pred_np, start_batch):
            z, y, x0 = st
            out[z : z + patch_size[0], y : y + patch_size[1], x0 : x0 + patch_size[2]] += p * imp
            weight[z : z + patch_size[0], y : y + patch_size[1], x0 : x0 + patch_size[2]] += imp
        patch_batch, coord_batch, start_batch = [], [], []

    for z in starts[0]:
        for y in starts[1]:
            for x0 in starts[2]:
                patch = vol[z : z + patch_size[0], y : y + patch_size[1], x0 : x0 + patch_size[2]]
                patch_batch.append(patch)
                if use_coords:
                    coord_batch.append(make_coord_channels(patch_size, starts=(z, y, x0), full_shape=full_shape))
                start_batch.append((z, y, x0))
                if len(patch_batch) >= sw_batch_size:
                    flush()
    flush()
    pred = out / np.maximum(weight, 1e-6)
    pred = pred[: original_shape[0], : original_shape[1], : original_shape[2]]
    return np.clip(pred, 0.0, 1.0).astype(np.float32, copy=False)
