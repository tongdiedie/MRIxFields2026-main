#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp.constants import FIELD_STRENGTH, MODALITIES, TASK_PAIRS
from mvp.data import write_manifest
from mvp.infer_utils import sliding_window_predict
from mvp.io import load_nifti, parse_mrix_filename, retrospective_paths, save_nifti
from mvp.networks import build_fds_from_config
from mvp.paths import env_path, load_env, resolve_path
from mvp.train_utils import device_from_arg, load_model_weights, load_yaml


def synthetic_name(original: Path, field: str) -> str:
    info = parse_mrix_filename(original)
    return f"S_{info['mod']}_{field}_{info['sid']}.nii.gz"


def main():
    ap = argparse.ArgumentParser(description="Generate anatomy-paired synthetic data with trained FDS.")
    ap.add_argument("--checkpoint", required=True, help="FDS best.pt/last.pt")
    ap.add_argument("--config", default=None, help="FDS config. If omitted, cfg inside checkpoint is used.")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--retro_split", default="Training_retrospective")
    ap.add_argument("--task", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--modalities", nargs="+", default=MODALITIES)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--limit_per_field", type=int, default=None)
    ap.add_argument("--patch_size", nargs=3, type=int, default=None)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--sw_batch_size", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    load_env(ROOT)
    data_root = resolve_path(args.data_root) if args.data_root else env_path("DATA_DIR", required=True)
    out_root = resolve_path(args.out_root)
    assert data_root is not None and out_root is not None
    out_root.mkdir(parents=True, exist_ok=True)

    # Load cfg from checkpoint unless overridden.
    raw = torch.load(str(resolve_path(args.checkpoint)), map_location="cpu")
    cfg = load_yaml(args.config) if args.config else raw.get("cfg", {})
    if not cfg:
        raise RuntimeError("No config found. Pass --config matching the FDS checkpoint.")
    device = device_from_arg(args.device or cfg.get("device"))
    model = build_fds_from_config(cfg).to(device)
    model.load_state_dict(raw.get("model", raw), strict=True)
    model.eval()

    patch_size = args.patch_size or cfg.get("patch_size", [96, 96, 96])
    use_coords = cfg.get("fds", {}).get("use_coords", True)
    rows = []
    pairs = TASK_PAIRS[args.task]

    for mod in args.modalities:
        for src_field, tgt_field in pairs:
            pair_name = f"{src_field}_to_{tgt_field}"
            src_v = FIELD_STRENGTH[src_field]
            tgt_v = FIELD_STRENGTH[tgt_field]

            if tgt_v > src_v:
                # Upfield training pair: real high target -> synthetic low source.
                real_field = tgt_field
                real_paths = retrospective_paths(data_root, mod, real_field, split=args.retro_split)
                if args.limit_per_field is not None:
                    real_paths = real_paths[: args.limit_per_field]
                for real_tgt in tqdm(real_paths, desc=f"synth source {mod} {pair_name}"):
                    arr, affine, header = load_nifti(real_tgt)
                    synth_src = sliding_window_predict(
                        model,
                        arr,
                        source_field=tgt_field,  # FDS from-field
                        target_field=src_field,  # FDS to-field
                        modality=mod,
                        patch_size=patch_size,
                        overlap=args.overlap,
                        sw_batch_size=args.sw_batch_size,
                        device=device,
                        amp=args.amp,
                        use_coords=use_coords,
                    )
                    out_path = out_root / mod / pair_name / "synthetic_source" / synthetic_name(real_tgt, src_field)
                    save_nifti(synth_src, affine, out_path, header=header)
                    sid = parse_mrix_filename(real_tgt)["sid"]
                    rows.append(
                        {
                            "source_path": str(out_path.resolve()),
                            "target_path": str(real_tgt.resolve()),
                            "source_field": src_field,
                            "target_field": tgt_field,
                            "modality": mod,
                            "subject_id": sid,
                            "is_synthetic": "1",
                        }
                    )
            else:
                # Downfield training pair: real high source -> synthetic low target.
                real_field = src_field
                real_paths = retrospective_paths(data_root, mod, real_field, split=args.retro_split)
                if args.limit_per_field is not None:
                    real_paths = real_paths[: args.limit_per_field]
                for real_src in tqdm(real_paths, desc=f"synth target {mod} {pair_name}"):
                    arr, affine, header = load_nifti(real_src)
                    synth_tgt = sliding_window_predict(
                        model,
                        arr,
                        source_field=src_field,
                        target_field=tgt_field,
                        modality=mod,
                        patch_size=patch_size,
                        overlap=args.overlap,
                        sw_batch_size=args.sw_batch_size,
                        device=device,
                        amp=args.amp,
                        use_coords=use_coords,
                    )
                    out_path = out_root / mod / pair_name / "synthetic_target" / synthetic_name(real_src, tgt_field)
                    save_nifti(synth_tgt, affine, out_path, header=header)
                    sid = parse_mrix_filename(real_src)["sid"]
                    rows.append(
                        {
                            "source_path": str(real_src.resolve()),
                            "target_path": str(out_path.resolve()),
                            "source_field": src_field,
                            "target_field": tgt_field,
                            "modality": mod,
                            "subject_id": sid,
                            "is_synthetic": "1",
                        }
                    )

    manifest = out_root / f"synthetic_manifest_task{args.task}.csv"
    write_manifest(manifest, rows)
    print(f"Wrote {len(rows)} synthetic paired rows: {manifest}")


if __name__ == "__main__":
    main()
