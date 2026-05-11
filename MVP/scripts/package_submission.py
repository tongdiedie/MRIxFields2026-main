#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp.constants import MODALITIES, TASK_PAIRS
from mvp.paths import resolve_path


def main():
    ap = argparse.ArgumentParser(description="Zip submission-style prediction tree.")
    ap.add_argument("--task", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--pred_root", required=True, help="Directory containing T1W/T2W/T2FLAIR subfolders")
    ap.add_argument("--zip_path", required=True)
    ap.add_argument("--allow_missing_seg", action="store_true", help="Do not warn if Task1/2 seg dirs are missing")
    args = ap.parse_args()

    root = resolve_path(args.pred_root)
    zip_path = resolve_path(args.zip_path)
    assert root is not None and zip_path is not None
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task in (1, 2) and not args.allow_missing_seg:
        missing = []
        for mod in MODALITIES:
            for src, tgt in TASK_PAIRS[args.task]:
                seg_dir = root / mod / f"{src}_to_{tgt}" / "seg"
                if not seg_dir.exists() or not list(seg_dir.glob("*.nii.gz")):
                    missing.append(str(seg_dir.relative_to(root)))
        if missing:
            print("WARNING: Task 1/2 require seg/ directories. Missing or empty examples:")
            for m in missing[:12]:
                print("  ", m)
            print("Use --allow_missing_seg only for local debugging, not for leaderboard submission.")

    files = []
    for mod in MODALITIES:
        d = root / mod
        if d.exists():
            files.extend(sorted(d.rglob("*.nii.gz")))
    if not files:
        raise RuntimeError(f"No .nii.gz files found under {root}/T1W,T2W,T2FLAIR")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in files:
            z.write(f, arcname=str(f.relative_to(root)))
    print(f"Wrote {zip_path} with {len(files)} files")


if __name__ == "__main__":
    main()
