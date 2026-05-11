#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp.constants import FIELD_STRENGTH, MODALITIES
from mvp.data import PairPatchDataset, collate_pair_batch, write_manifest
from mvp.io import prospective_pair_paths
from mvp.losses import MVPReconstructionLoss
from mvp.networks import build_fds_from_config
from mvp.paths import env_path, load_env, resolve_path
from mvp.train_utils import device_from_arg, load_yaml, save_checkpoint, save_yaml, set_seed


def default_downfield_pairs():
    fields = sorted(FIELD_STRENGTH, key=lambda f: FIELD_STRENGTH[f])
    pairs = []
    for hi in fields:
        for lo in fields:
            if FIELD_STRENGTH[hi] > FIELD_STRENGTH[lo]:
                pairs.append((hi, lo))
    return pairs


def parse_pair_list(items):
    if not items:
        return default_downfield_pairs()
    pairs = []
    for item in items:
        if ":" in item:
            a, b = item.split(":", 1)
        elif "_to_" in item:
            a, b = item.split("_to_", 1)
        else:
            raise ValueError(f"Bad pair '{item}'. Use 7T:0.1T or 7T_to_0.1T")
        pairs.append((a.strip(), b.strip()))
    return pairs


def build_fds_rows(data_root, split, modalities, pairs, max_subjects=None):
    rows = []
    for mod in modalities:
        for from_field, to_field in pairs:
            matched = prospective_pair_paths(data_root, split, mod, from_field, to_field)
            if max_subjects is not None:
                matched = matched[:max_subjects]
            for sid, from_path, to_path in matched:
                rows.append(
                    {
                        "source_path": str(from_path.resolve()),
                        "target_path": str(to_path.resolve()),
                        "source_field": from_field,
                        "target_field": to_field,
                        "modality": mod,
                        "subject_id": sid,
                        "is_synthetic": "0",
                    }
                )
            print(f"FDS {mod} {from_field}->{to_field}: {len(matched)} pairs")
    return rows


def run_epoch(model, loader, criterion, optimizer, device, amp=False):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    scaler = run_epoch.scaler if train else None
    pbar = tqdm(loader, desc="train" if train else "valid", leave=False)
    for batch in pbar:
        x = batch["source"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        coords = batch.get("coords")
        if coords is not None:
            coords = coords.to(device, non_blocking=True)
        enabled_amp = bool(amp and device.type == "cuda")
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=enabled_amp):
                pred = model(x, batch["source_field"], batch["target_field"], batch["modality"], coords=coords)
                loss, logs = criterion(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if enabled_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        bs = x.shape[0]
        total += float(loss.detach().cpu()) * bs
        n += bs
        pbar.set_postfix(loss=total / max(n, 1), **{k: f"{v:.4f}" for k, v in logs.items() if k != "total"})
    return total / max(n, 1)


run_epoch.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser(description="Train field degradation simulator (high/source -> low/target).")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "fds_all_downfield.yaml"))
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--pairs", nargs="*", default=None, help="Override pairs, e.g. 7T:0.1T 7T:1.5T")
    ap.add_argument("--modalities", nargs="+", default=None)
    ap.add_argument("--max_subjects", type=int, default=None)
    args = ap.parse_args()

    load_env(ROOT)
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 1234)))
    data_root = resolve_path(args.data_root) if args.data_root else env_path("DATA_DIR", required=True)
    output_dir = resolve_path(args.output_dir or cfg.get("output_dir", "./runs/fds"))
    assert data_root is not None and output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output_dir / "config.yaml")

    modalities = args.modalities or cfg.get("modalities", MODALITIES)
    pairs = parse_pair_list(args.pairs) if args.pairs else [tuple(x) for x in cfg.get("fds_pairs", default_downfield_pairs())]
    max_subjects = args.max_subjects if args.max_subjects is not None else cfg.get("max_subjects")

    train_rows = build_fds_rows(data_root, cfg.get("train_split", "Training_prospective"), modalities, pairs, max_subjects)
    train_manifest = output_dir / "fds_train_manifest.csv"
    write_manifest(train_manifest, train_rows)
    val_manifest = None
    val_split = cfg.get("val_split")
    if val_split:
        val_rows = build_fds_rows(data_root, val_split, modalities, pairs, cfg.get("max_val_subjects"))
        val_manifest = output_dir / "fds_val_manifest.csv"
        write_manifest(val_manifest, val_rows)

    device = device_from_arg(args.device or cfg.get("device"))
    model = build_fds_from_config(cfg).to(device)
    train_ds = PairPatchDataset(
        [train_manifest],
        patch_size=cfg.get("patch_size", [96, 96, 96]),
        samples_per_volume=cfg.get("samples_per_volume", 4),
        cache_items=cfg.get("cache_items", 4),
        use_coords=cfg.get("fds", {}).get("use_coords", True),
        foreground_prob=cfg.get("foreground_prob", 0.7),
        seed=cfg.get("seed", 1234),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_pair_batch,
    )
    val_loader = None
    if val_manifest:
        val_ds = PairPatchDataset(
            [val_manifest],
            patch_size=cfg.get("patch_size", [96, 96, 96]),
            samples_per_volume=cfg.get("val_samples_per_volume", 1),
            cache_items=cfg.get("cache_items", 4),
            use_coords=cfg.get("fds", {}).get("use_coords", True),
            foreground_prob=cfg.get("foreground_prob", 0.7),
            seed=cfg.get("seed", 1234) + 99,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 2), collate_fn=collate_pair_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-4), weight_decay=cfg.get("weight_decay", 1e-4))
    criterion = MVPReconstructionLoss(cfg.get("loss", {"l1": 1.0, "nrmse": 0.2, "ssim": 0.1, "lpips": 0.0}))
    best = float("inf")
    epochs = int(cfg.get("epochs", 50))
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, amp=cfg.get("amp", True))
        val_loss = train_loss
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, criterion, None, device, amp=False)
        print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, cfg, extra={"val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, cfg, extra={"val_loss": val_loss})
            print(f"Saved best.pt ({best:.5f})")


if __name__ == "__main__":
    main()
