from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .constants import FIELDS, MODALITIES

_NIFTI_RE = re.compile(r"^(?P<prefix>[RP])_(?P<mod>T1W|T2W|T2FLAIR)_(?P<field>0\.1T|1\.5T|3T|5T|7T)_(?P<sid>\d{4})\.nii\.gz$")


def parse_mrix_filename(path: str | Path) -> Dict[str, str]:
    name = Path(path).name
    m = _NIFTI_RE.match(name)
    if not m:
        raise ValueError(f"Not an MRIxFields filename: {name}")
    return m.groupdict()


def replace_field_in_name(path: str | Path, target_field: str, prefix: Optional[str] = None) -> str:
    info = parse_mrix_filename(path)
    pref = prefix if prefix is not None else info["prefix"]
    return f"{pref}_{info['mod']}_{target_field}_{info['sid']}.nii.gz"


def add_seg_suffix(name: str | Path) -> str:
    s = Path(name).name
    if not s.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz name, got {s}")
    return s[:-7] + "_seg.nii.gz"


def find_niftis(root: str | Path) -> List[Path]:
    return sorted(Path(root).rglob("*.nii.gz"))


def index_by_subject(field_dir: str | Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in find_niftis(field_dir):
        try:
            info = parse_mrix_filename(p)
        except ValueError:
            continue
        out[info["sid"]] = p
    return out


def load_nifti(path: str | Path) -> Tuple[np.ndarray, object, object]:
    """Load NIfTI as float32 and return (array, affine, header).

    The challenge data are already in MNI space and [0,1]. We preserve the
    original orientation/header for saving predictions. This function clips only
    when saving, not when loading, so training can inspect outliers if any.
    """
    import nibabel as nib

    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    return arr.astype(np.float32, copy=False), img.affine, img.header.copy()


def save_nifti(data: np.ndarray, affine, out_path: str | Path, header=None, clip: bool = True) -> None:
    import nibabel as nib

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = data.astype(np.float32, copy=False)
    if clip:
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    if header is not None:
        header = header.copy()
        header.set_data_dtype(np.float32)
    img = nib.Nifti1Image(arr, affine, header=header)
    nib.save(img, str(out_path))


def prospective_pair_paths(
    data_root: str | Path,
    split: str,
    modality: str,
    source_field: str,
    target_field: str,
) -> List[Tuple[str, Path, Path]]:
    """Return subject-id matched source/target paths from a prospective split."""
    root = Path(data_root) / split / modality
    src_dir = root / source_field
    tgt_dir = root / target_field
    src_idx = index_by_subject(src_dir)
    tgt_idx = index_by_subject(tgt_dir)
    sids = sorted(set(src_idx).intersection(tgt_idx))
    return [(sid, src_idx[sid], tgt_idx[sid]) for sid in sids]


def retrospective_paths(
    data_root: str | Path,
    modality: str,
    field: str,
    split: str = "Training_retrospective",
) -> List[Path]:
    return find_niftis(Path(data_root) / split / modality / field)


def make_pair_name(source_field: str, target_field: str) -> str:
    return f"{source_field}_to_{target_field}"
