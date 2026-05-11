#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp.constants import MODALITIES, TASK_PAIRS
from mvp.io import prospective_pair_paths
from mvp.data import write_manifest
from mvp.paths import env_path, resolve_path, load_env


def parse_pairs(raw: str):
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            s, t = item.split(":", 1)
        elif "_to_" in item:
            s, t = item.split("_to_", 1)
        else:
            raise ValueError(f"Cannot parse pair '{item}'. Use 0.1T:7T or 0.1T_to_7T")
        out.append((s.strip(), t.strip()))
    return out


def main():
    p = argparse.ArgumentParser(description="Build real paired prospective manifest for MRIxFields MVP.")
    p.add_argument("--data_root", type=str, default=None, help="Dataset root. Defaults to DATA_DIR from .env")
    p.add_argument("--split", type=str, default="Training_prospective")
    p.add_argument("--task", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--pairs", type=str, default=None, help="Optional comma list, e.g. '0.1T:7T,1.5T:7T'")
    p.add_argument("--modalities", nargs="+", default=MODALITIES)
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--out_csv", type=str, required=True)
    args = p.parse_args()

    load_env(ROOT)
    data_root = resolve_path(args.data_root) if args.data_root else env_path("DATA_DIR", required=True)
    assert data_root is not None
    pairs = parse_pairs(args.pairs) if args.pairs else TASK_PAIRS[args.task]

    rows = []
    for mod in args.modalities:
        for src, tgt in pairs:
            matched = prospective_pair_paths(data_root, args.split, mod, src, tgt)
            if args.max_subjects is not None:
                matched = matched[: args.max_subjects]
            for sid, src_path, tgt_path in matched:
                rows.append(
                    {
                        "source_path": str(src_path.resolve()),
                        "target_path": str(tgt_path.resolve()),
                        "source_field": src,
                        "target_field": tgt,
                        "modality": mod,
                        "subject_id": sid,
                        "is_synthetic": "0",
                    }
                )
            print(f"{mod} {src}->{tgt}: {len(matched)} real pairs")
    out_csv = resolve_path(args.out_csv)
    assert out_csv is not None
    write_manifest(out_csv, rows)
    print(f"Wrote {len(rows)} rows: {out_csv}")


if __name__ == "__main__":
    main()
