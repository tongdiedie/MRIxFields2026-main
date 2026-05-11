#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def main():
    ap = argparse.ArgumentParser(description="Wrapper around official Evaluation/evaluate.py")
    ap.add_argument("--task", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--pred_seg_dir", default=None)
    ap.add_argument("--target_seg_dir", default=None)
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    eval_py = REPO_ROOT / "Evaluation" / "evaluate.py"
    if not eval_py.exists():
        raise RuntimeError(f"Cannot find official evaluator at {eval_py}. Place MVP/ at repo root.")
    metrics = ["nrmse", "ssim", "lpips"] if args.task == 3 else ["nrmse", "ssim", "lpips", "dice", "volume"]
    cmd = [
        sys.executable,
        str(eval_py),
        "--pred_dir",
        args.pred_dir,
        "--target_dir",
        args.target_dir,
        "--metrics",
        *metrics,
        "--device",
        args.device,
    ]
    if args.output_csv:
        cmd += ["--output_csv", args.output_csv]
    if args.output_json:
        cmd += ["--output_json", args.output_json]
    if args.task in (1, 2):
        if not args.pred_seg_dir or not args.target_seg_dir:
            raise RuntimeError("Task 1/2 official eval requires --pred_seg_dir and --target_seg_dir")
        cmd += ["--pred_seg_dir", args.pred_seg_dir, "--target_seg_dir", args.target_seg_dir]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
