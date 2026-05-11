from __future__ import annotations

import csv
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_nifti
from .networks import make_coord_channels


@dataclass
class PairRow:
    source_path: str
    target_path: str
    source_field: str
    target_field: str
    modality: str
    subject_id: str = ""
    is_synthetic: str = "0"


def read_manifest(path: str | Path) -> List[PairRow]:
    rows: List[PairRow] = []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_path", "target_path", "source_field", "target_field", "modality"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
        for r in reader:
            rows.append(
                PairRow(
                    source_path=r["source_path"],
                    target_path=r["target_path"],
                    source_field=r["source_field"],
                    target_field=r["target_field"],
                    modality=r["modality"],
                    subject_id=r.get("subject_id", ""),
                    is_synthetic=r.get("is_synthetic", "0"),
                )
            )
    if not rows:
        raise ValueError(f"Manifest {path} contains no rows")
    return rows


def write_manifest(path: str | Path, rows: Iterable[Dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_path", "target_path", "source_field", "target_field", "modality", "subject_id", "is_synthetic"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


class VolumeCache:
    def __init__(self, max_items: int = 4):
        self.max_items = int(max_items)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str | Path) -> np.ndarray:
        key = str(path)
        if key in self._cache:
            arr = self._cache.pop(key)
            self._cache[key] = arr
            return arr
        arr, _, _ = load_nifti(path)
        arr = arr.astype(np.float32, copy=False)
        self._cache[key] = arr
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return arr


def _pad_to_shape(arr: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    pads = []
    for n, want in zip(arr.shape, shape):
        extra = max(int(want) - int(n), 0)
        before = extra // 2
        after = extra - before
        pads.append((before, after))
    if any(b or a for b, a in pads):
        arr = np.pad(arr, pads, mode="constant", constant_values=0)
    return arr


def _random_starts(shape: Sequence[int], patch: Sequence[int], rng: random.Random) -> Tuple[int, int, int]:
    return tuple(rng.randint(0, max(int(n) - int(p), 0)) for n, p in zip(shape, patch))  # type: ignore[return-value]


def _crop(arr: np.ndarray, starts: Sequence[int], patch: Sequence[int]) -> np.ndarray:
    z, y, x = starts
    dz, dy, dx = patch
    return arr[z : z + dz, y : y + dy, x : x + dx]


class PairPatchDataset(Dataset):
    """Random 3D patch dataset from a paired manifest."""

    def __init__(
        self,
        manifest_paths: Sequence[str | Path],
        patch_size: Sequence[int] = (96, 96, 96),
        samples_per_volume: int = 4,
        cache_items: int = 4,
        foreground_prob: float = 0.7,
        foreground_threshold: float = 1e-4,
        max_foreground_tries: int = 16,
        use_coords: bool = True,
        seed: int = 1234,
    ):
        self.rows: List[PairRow] = []
        for p in manifest_paths:
            self.rows.extend(read_manifest(p))
        self.patch_size = tuple(int(v) for v in patch_size)
        if len(self.patch_size) != 3:
            raise ValueError("patch_size must be [D,H,W]")
        self.samples_per_volume = int(samples_per_volume)
        self.cache = VolumeCache(max_items=cache_items)
        self.foreground_prob = float(foreground_prob)
        self.foreground_threshold = float(foreground_threshold)
        self.max_foreground_tries = int(max_foreground_tries)
        self.use_coords = bool(use_coords)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.rows) * self.samples_per_volume

    def _choose_patch(self, src: np.ndarray, tgt: np.ndarray, rng: random.Random):
        if src.shape != tgt.shape:
            raise ValueError(f"Source/target shapes differ: {src.shape} vs {tgt.shape}")
        src = _pad_to_shape(src, self.patch_size)
        tgt = _pad_to_shape(tgt, self.patch_size)
        shape = src.shape
        starts = _random_starts(shape, self.patch_size, rng)
        # Avoid wasting most patches on pure background.
        if rng.random() < self.foreground_prob:
            for _ in range(self.max_foreground_tries):
                cand = _random_starts(shape, self.patch_size, rng)
                patch = _crop(tgt, cand, self.patch_size)
                if float(patch.mean()) > self.foreground_threshold:
                    starts = cand
                    break
        return _crop(src, starts, self.patch_size), _crop(tgt, starts, self.patch_size), starts, shape

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx % len(self.rows)]
        rng = random.Random(self.seed + idx + random.randint(0, 10_000_000))
        src = self.cache.get(row.source_path)
        tgt = self.cache.get(row.target_path)
        src_p, tgt_p, starts, full_shape = self._choose_patch(src, tgt, rng)
        src_t = torch.from_numpy(src_p[None].astype(np.float32, copy=False))
        tgt_t = torch.from_numpy(tgt_p[None].astype(np.float32, copy=False))
        item: Dict[str, object] = {
            "source": src_t,
            "target": tgt_t,
            "source_field": row.source_field,
            "target_field": row.target_field,
            "modality": row.modality,
            "subject_id": row.subject_id,
        }
        if self.use_coords:
            item["coords"] = make_coord_channels(self.patch_size, starts=starts, full_shape=full_shape)
        return item


def collate_pair_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["source"] = torch.stack([b["source"] for b in batch])  # type: ignore[list-item]
    out["target"] = torch.stack([b["target"] for b in batch])  # type: ignore[list-item]
    if "coords" in batch[0]:
        out["coords"] = torch.stack([b["coords"] for b in batch])  # type: ignore[list-item]
    out["source_field"] = [str(b["source_field"]) for b in batch]
    out["target_field"] = [str(b["target_field"]) for b in batch]
    out["modality"] = [str(b["modality"]) for b in batch]
    out["subject_id"] = [str(b.get("subject_id", "")) for b in batch]
    return out
