#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp.data import PairPatchDataset, collate_pair_batch
from mvp.losses import MVPReconstructionLoss
from mvp.networks import build_model_from_config
from mvp.paths import load_env, resolve_path
from mvp.train_utils import device_from_arg, load_model_weights, load_yaml, save_checkpoint, save_yaml, set_seed


def run_epoch(model, loader, criterion, optimizer, device, amp=False):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    logs_sum = {}
    pbar = tqdm(loader, desc="train" if is_train else "valid", leave=False)
    for batch in pbar:
        src = batch["source"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        coords = batch.get("coords")
        if coords is not None:
            coords = coords.to(device, non_blocking=True)
        enabled_amp = bool(amp and device.type == "cuda")
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=enabled_amp):
                pred = model(src, batch["source_field"], batch["target_field"], batch["modality"], coords=coords)
                loss, logs = criterion(pred, tgt)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if enabled_amp:
                    run_epoch.scaler.scale(loss).backward()
                    run_epoch.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    run_epoch.scaler.step(optimizer)
                    run_epoch.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        bs = src.shape[0]
        total += float(loss.detach().cpu()) * bs
        count += bs
        for k, v in logs.items():
            logs_sum[k] = logs_sum.get(k, 0.0) + float(v) * bs
        pbar.set_postfix(loss=total / max(count, 1), **{k: f"{logs_sum[k]/max(count,1):.4f}" for k in logs_sum if k != "total"})
    avg_logs = {k: v / max(count, 1) for k, v in logs_sum.items()}
    avg_logs["loss"] = total / max(count, 1)
    return avg_logs


run_epoch.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser(description="Train 3D/2.5D field-conditioned U-Net translator.")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "task1_mvp.yaml"))
    ap.add_argument("--manifest", nargs="+", default=None, help="Training manifest CSV(s). Overrides config train_manifests.")
    ap.add_argument("--val_manifest", nargs="+", default=None, help="Validation manifest CSV(s).")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    load_env(ROOT)
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 1234)))
    output_dir = resolve_path(args.output_dir or cfg.get("output_dir", "./runs/translator_task1"))
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output_dir / "config.yaml")

    train_manifests = args.manifest or cfg.get("train_manifests")
    if not train_manifests:
        raise RuntimeError("Provide --manifest or train_manifests in config.")
    train_manifests = [str(resolve_path(p)) for p in train_manifests]
    val_manifests = args.val_manifest or cfg.get("val_manifests")
    val_manifests = [str(resolve_path(p)) for p in val_manifests] if val_manifests else []

    device = device_from_arg(args.device or cfg.get("device"))
    model = build_model_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 2e-4), weight_decay=cfg.get("weight_decay", 1e-4)
    )
    start_epoch = 1
    if args.resume:
        ckpt = load_model_weights(model, resolve_path(args.resume), strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    dc = cfg.get("data", {})
    train_ds = PairPatchDataset(
        train_manifests,
        patch_size=dc.get("patch_size", [96, 96, 96]),
        samples_per_volume=dc.get("samples_per_volume", 4),
        cache_items=dc.get("cache_items", 4),
        foreground_prob=dc.get("foreground_prob", 0.7),
        use_coords=cfg.get("model", {}).get("use_coords", True),
        seed=cfg.get("seed", 1234),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_pair_batch,
        drop_last=False,
    )
    val_loader = None
    if val_manifests:
        val_ds = PairPatchDataset(
            val_manifests,
            patch_size=dc.get("patch_size", [96, 96, 96]),
            samples_per_volume=dc.get("val_samples_per_volume", 1),
            cache_items=dc.get("cache_items", 4),
            foreground_prob=dc.get("foreground_prob", 0.7),
            use_coords=cfg.get("model", {}).get("use_coords", True),
            seed=cfg.get("seed", 1234) + 202,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, cfg.get("num_workers", 4) // 2),
            pin_memory=True,
            collate_fn=collate_pair_batch,
        )

    criterion = MVPReconstructionLoss(
        cfg.get("loss", {"l1": 1.0, "nrmse": 0.5, "ssim": 0.2, "lpips": 0.02}),
        lpips_axis=cfg.get("loss_slice_axis", 2),
        lpips_net=cfg.get("lpips_net", "alex"),
    )
    best = float("inf")
    log_path = output_dir / "log.csv"
    first_log = not log_path.exists()
    epochs = int(cfg.get("epochs", 100))
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_logs = run_epoch(model, train_loader, criterion, optimizer, device, amp=cfg.get("amp", True))
        val_logs = {}
        score = train_logs["loss"]
        if val_loader is not None:
            with torch.no_grad():
                val_logs = run_epoch(model, val_loader, criterion, None, device, amp=False)
                score = val_logs.get("nrmse", val_logs["loss"])
        print(f"train: {train_logs}")
        if val_logs:
            print(f"valid: {val_logs}")

        with log_path.open("a", newline="") as f:
            keys = ["epoch"] + [f"train_{k}" for k in train_logs] + [f"val_{k}" for k in val_logs]
            writer = csv.DictWriter(f, fieldnames=keys)
            if first_log:
                writer.writeheader()
                first_log = False
            row = {"epoch": epoch}
            row.update({f"train_{k}": v for k, v in train_logs.items()})
            row.update({f"val_{k}": v for k, v in val_logs.items()})
            writer.writerow(row)

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, cfg, extra={"score": score})
        if score < best:
            best = score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, cfg, extra={"score": score})
            print(f"Saved best.pt with score={best:.5f}")
        if epoch % int(cfg.get("save_every", 10)) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, cfg, extra={"score": score})


if __name__ == "__main__":
    main()
