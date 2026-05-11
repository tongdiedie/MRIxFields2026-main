#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp.infer_utils import sliding_window_predict
from mvp.io import load_nifti, parse_mrix_filename, save_nifti
from mvp.networks import build_model_from_config
from mvp.paths import load_env, resolve_path
from mvp.train_utils import device_from_arg, load_yaml


def output_name(input_path: Path, target_field: str) -> str:
    info = parse_mrix_filename(input_path)
    # For validation/test inputs prefix is P. Keep the prefix; scorer expects P.
    return f"{info['prefix']}_{info['mod']}_{target_field}_{info['sid']}.nii.gz"


def main():
    ap = argparse.ArgumentParser(description="Sliding-window inference for MVP field-conditioned U-Net.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="If omitted, cfg saved inside checkpoint is used.")
    ap.add_argument("--input_dir", required=True, help="Directory containing source-field .nii.gz files")
    ap.add_argument("--output_dir", required=True, help="Root for submission-style outputs")
    ap.add_argument("--source_field", required=True)
    ap.add_argument("--target_field", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--patch_size", nargs=3, type=int, default=None)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--sw_batch_size", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--flat", action="store_true", help="Save directly under output_dir instead of modality/pair/pred")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    load_env(ROOT)
    ckpt_path = resolve_path(args.checkpoint)
    assert ckpt_path is not None
    raw = torch.load(str(ckpt_path), map_location="cpu")
    cfg = load_yaml(args.config) if args.config else raw.get("cfg", {})
    if not cfg:
        raise RuntimeError("No config found in checkpoint. Pass --config.")

    device = device_from_arg(args.device or cfg.get("device"))
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(raw.get("model", raw), strict=True)
    model.eval()

    input_dir = resolve_path(args.input_dir)
    output_root = resolve_path(args.output_dir)
    assert input_dir is not None and output_root is not None
    pair = f"{args.source_field}_to_{args.target_field}"
    out_dir = output_root if args.flat else output_root / args.modality / pair / "pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_size = args.patch_size or cfg.get("data", {}).get("patch_size", [96, 96, 96])
    use_coords = cfg.get("model", {}).get("use_coords", True)
    files = sorted(input_dir.rglob("*.nii.gz"))
    if not files:
        raise RuntimeError(f"No .nii.gz files found in {input_dir}")
    for p in tqdm(files, desc=f"infer {args.modality} {pair}"):
        out_path = out_dir / output_name(p, args.target_field)
        if out_path.exists() and not args.overwrite:
            continue
        vol, affine, header = load_nifti(p)
        pred = sliding_window_predict(
            model,
            vol,
            source_field=args.source_field,
            target_field=args.target_field,
            modality=args.modality,
            patch_size=patch_size,
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size,
            device=device,
            amp=args.amp,
            use_coords=use_coords,
        )
        save_nifti(pred, affine, out_path, header=header, clip=True)
    print(f"Saved predictions to {out_dir}")


if __name__ == "__main__":
    main()
