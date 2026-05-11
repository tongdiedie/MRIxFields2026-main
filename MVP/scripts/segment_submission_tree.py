#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def main():
    ap = argparse.ArgumentParser(description="Run official SynthSeg script for every submission pred/ directory.")
    ap.add_argument("--submission_root", required=True, help="Root containing T1W/<pair>/pred")
    ap.add_argument("--task", type=int, choices=[1, 2], required=True, help="Only Task 1/2 need segmentation")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    seg_py = REPO_ROOT / "Evaluation" / "segment.py"
    if not seg_py.exists():
        raise RuntimeError(f"Cannot find official segment.py at {seg_py}. Place MVP/ at repo root.")
    root = Path(args.submission_root)
    pred_dirs = sorted([p for p in root.rglob("pred") if p.is_dir()])
    if not pred_dirs:
        raise RuntimeError(f"No pred/ directories found under {root}")
    for pred_dir in pred_dirs:
        seg_dir = pred_dir.parent / "seg"
        if seg_dir.exists() and list(seg_dir.glob("*.nii.gz")) and not args.overwrite:
            print(f"Skipping existing {seg_dir}")
            continue
        seg_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(seg_py), "--input_dir", str(pred_dir), "--output_dir", str(seg_dir)]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
