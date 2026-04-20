"""Unified training script for MRIxFields2026 baselines.

Supports:
    - CUT (per-pair unpaired training)
    - CycleGAN (per-pair unpaired training)
    - StarGAN v2 (multi-domain training)
    - Hybrid (unpaired pretrain + paired finetune)

Usage:
    # Single GPU
    python scripts/train.py --config configs/task1/cyclegan/0.1T_to_7T_T1W.yaml --mode retro_scratch

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=2 scripts/train.py --config configs/task1/cyclegan/0.1T_to_7T_T1W.yaml --mode retro_scratch --dist
"""

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.models.cut_model import CUTModel
from mrixfields.models.cyclegan_model import CycleGANModel
from mrixfields.models.stargan_v2 import (
    build_stargan_v2, compute_d_loss, compute_g_loss, moving_average,
)
from mrixfields.models.networks import ResnetGenerator, init_net
from mrixfields.data.dataset import UnpairedMRIDataset, PairedMRIDataset, MultiDomainMRIDataset
from mrixfields.data.cached_dataset import CachedUnpairedDataset, CachedPairedDataset, CachedMultiDomainDataset
from mrixfields.data.unpaired_loader import UnpairedDataLoader
from mrixfields.losses.perceptual import PerceptualLoss
from mrixfields.losses.structure import SSIMLoss, StructureLoss
from mrixfields.data.utils import ABBR_TO_SPLIT as SPLIT_MAP
from mrixfields.env import load_env, get_data_dir, get_preprocessed_dir, get_output_dir, get_device
from mrixfields.utils_dist import init_dist, get_dist_info, is_main_process, cleanup_dist
# SPLIT_MAP: yaml config "split" value → actual data directory name.
# split is the data subset (independent from --mode which controls pipeline stage).


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_lr_lambda(n_epochs: int, n_epochs_decay: int):
    """Linear LR decay lambda: constant for n_epochs, then linear decay to 0."""
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
    return lambda_rule


# --------------------------------------------------------------------------- #
#  CUT Training
# --------------------------------------------------------------------------- #

def _make_unpaired_loaders(dataset_src, dataset_tgt, batch_size, num_workers, use_dist):
    """Create unpaired data loaders, with DistributedSampler when distributed."""
    if use_dist:
        sampler_src = DistributedSampler(dataset_src, shuffle=True)
        sampler_tgt = DistributedSampler(dataset_tgt, shuffle=True)
        loader_src = DataLoader(
            dataset_src, batch_size=batch_size, sampler=sampler_src,
            num_workers=num_workers, drop_last=True, pin_memory=True,
        )
        loader_tgt = DataLoader(
            dataset_tgt, batch_size=batch_size, sampler=sampler_tgt,
            num_workers=num_workers, drop_last=True, pin_memory=True,
        )
        return loader_src, loader_tgt, sampler_src, sampler_tgt
    else:
        loader = UnpairedDataLoader(
            dataset_src, dataset_tgt,
            batch_size=batch_size, num_workers=num_workers,
        )
        return loader, None, None, None


def _iter_unpaired(loader_or_tuple, use_dist):
    """Iterate unpaired loaders. Yields (batch_a, batch_b)."""
    if use_dist:
        loader_src, loader_tgt = loader_or_tuple
        iter_src = iter(loader_src)
        iter_tgt = iter(loader_tgt)
        for _ in range(min(len(loader_src), len(loader_tgt))):
            batch_a = next(iter_src)
            batch_b = next(iter_tgt)
            yield batch_a, batch_b
    else:
        yield from loader_or_tuple


def train_cut(cfg: dict, device: torch.device, use_dist: bool = False):
    """Train CUT model for a specific field-strength pair."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    rank, world_size = get_dist_info()
    output_dir = Path(train_cfg["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = CUTModel(
        input_nc=model_cfg.get("input_nc", 1),
        output_nc=model_cfg.get("output_nc", 1),
        ngf=model_cfg.get("ngf", 64),
        ndf=model_cfg.get("ndf", 64),
        n_blocks=model_cfg.get("n_blocks", 9),
        nce_layers=model_cfg.get("nce_layers", [0, 4, 8, 12, 16]),
        nce_T=model_cfg.get("nce_T", 0.07),
        num_patches=model_cfg.get("num_patches", 256),
        lambda_GAN=model_cfg.get("lambda_GAN", 1.0),
        lambda_NCE=model_cfg.get("lambda_NCE", 1.0),
        nce_idt=model_cfg.get("nce_idt", True),
        gan_mode=model_cfg.get("gan_mode", "lsgan"),
        netF_nc=model_cfg.get("netF_nc", 256),
        lr=train_cfg.get("lr", 0.0002),
        beta1=train_cfg.get("beta1", 0.5),
        beta2=train_cfg.get("beta2", 0.999),
    ).to(device)
    model.init_weights()

    # Build datasets
    modality = data_cfg["modalities"][0]
    source_field = data_cfg["source_field"]
    target_field = data_cfg["target_field"]
    crop_size_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    split = train_cfg.get("split", "retro_train")

    if preprocessed_dir:
        dataset_src = CachedUnpairedDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modality=modality, field_strength=source_field, crop_size=crop_size,
        )
        dataset_tgt = CachedUnpairedDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modality=modality, field_strength=target_field, crop_size=crop_size,
        )
    else:
        split_full = SPLIT_MAP.get(split, split)
        dataset_src = UnpairedMRIDataset(
            data_root=data_cfg["data_dir"], split=split_full,
            modality=modality, field_strength=source_field, crop_size=crop_size,
        )
        dataset_tgt = UnpairedMRIDataset(
            data_root=data_cfg["data_dir"], split=split_full,
            modality=modality, field_strength=target_field, crop_size=crop_size,
        )

    batch_size = train_cfg.get("batch_size", 4)
    if use_dist:
        batch_size = batch_size // world_size
    num_workers = train_cfg.get("num_workers", 4)

    result = _make_unpaired_loaders(dataset_src, dataset_tgt, batch_size, num_workers, use_dist)
    if use_dist:
        loader_src, loader_tgt, sampler_src, sampler_tgt = result
        loader_len = min(len(loader_src), len(loader_tgt))
    else:
        loader = result[0]
        loader_len = len(loader)

    # Data-dependent init (before DDP wrapping)
    if use_dist:
        init_batch_a = next(iter(DataLoader(dataset_src, batch_size=batch_size)))
        init_batch_b = next(iter(DataLoader(dataset_tgt, batch_size=batch_size)))
    else:
        for batch_a, batch_b in loader:
            init_batch_a, init_batch_b = batch_a, batch_b
            break
    model.data_dependent_initialize(
        init_batch_a["image"].to(device), init_batch_b["image"].to(device)
    )
    model.setup_optimizers()

    # DDP wrap sub-networks
    if use_dist:
        model.netG = DDP(model.netG, device_ids=[device])
        model.netD = DDP(model.netD, device_ids=[device])
        model.netF = DDP(model.netF, device_ids=[device], find_unused_parameters=True)

    # LR schedulers
    n_epochs = train_cfg.get("n_epochs", 100)
    n_epochs_decay = train_cfg.get("n_epochs_decay", 100)
    lr_lambda = get_lr_lambda(n_epochs, n_epochs_decay)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(model.optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(model.optimizer_D, lr_lambda)
    scheduler_F = torch.optim.lr_scheduler.LambdaLR(model.optimizer_F, lr_lambda)

    total_epochs = n_epochs + n_epochs_decay
    max_iters = train_cfg.get("max_iters_per_epoch", 0)
    if is_main_process():
        print(f"Training CUT: {source_field} -> {target_field}, "
              f"{total_epochs} epochs, {loader_len} iters/epoch"
              + (f" (capped at {max_iters})" if max_iters else "")
              + (f" (DDP: {world_size} GPUs)" if use_dist else ""))

    for epoch in range(total_epochs):
        model.train()
        if use_dist:
            sampler_src.set_epoch(epoch)
            sampler_tgt.set_epoch(epoch)
            iterable = _iter_unpaired((loader_src, loader_tgt), True)
        else:
            iterable = loader

        pbar = tqdm(iterable, desc=f"Epoch {epoch+1}/{total_epochs}",
                    total=loader_len, disable=not is_main_process())
        for i, (batch_a, batch_b) in enumerate(pbar):
            if max_iters and i >= max_iters:
                break
            real_A = batch_a["image"].to(device)
            real_B = batch_b["image"].to(device)
            losses = model.optimize_parameters(real_A, real_B)
            pbar.set_postfix(G=f"{losses['loss_G']:.4f}", D=f"{losses['loss_D']:.4f}")

        scheduler_G.step()
        scheduler_D.step()
        scheduler_F.step()

        if is_main_process() and (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            ckpt = weights_dir / f"checkpoint_epoch{epoch+1}.pth"
            net_G = model.netG.module if use_dist else model.netG
            torch.save({"epoch": epoch + 1, "model": model.state_dict()}, ckpt)
            print(f"Saved: {ckpt}")

    if is_main_process():
        net_G = model.netG.module if use_dist else model.netG
        torch.save(net_G.state_dict(), weights_dir / "generator_final.pth")
        print(f"CUT training complete. Saved to: {output_dir}")


# --------------------------------------------------------------------------- #
#  CUT Fine-tuning (paired, supervised)
# --------------------------------------------------------------------------- #

def train_cut_finetune(cfg: dict, device: torch.device, use_dist: bool = False):
    """Phase 2 for CUT: fine-tune the pretrained ResnetGenerator with paired data.

    Discards CUT-specific structure (PatchNCE, discriminator, MLP head F) and
    keeps only the ResnetGenerator. Supervised with paired (source, target)
    images using L1 + LPIPS (+ optional SSIM, Edge) reconstruction loss.

    Used by --mode pro_pretrained (load CUT pretrain ckpt) and
    --mode pro_scratch (random init).
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    pretrain_cfg = cfg.get("pretrain", {})
    finetune_cfg = cfg.get("finetune", {})
    rank, world_size = get_dist_info()
    output_dir = Path(train_cfg["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    # Build generator (same ResnetGenerator as CUT pretrain)
    generator = ResnetGenerator(
        input_nc=model_cfg.get("input_nc", 1),
        output_nc=model_cfg.get("output_nc", 1),
        ngf=model_cfg.get("ngf", 64),
        n_blocks=model_cfg.get("n_blocks", 9),
    ).to(device)

    # Load pretrained weights
    ckpt_path = pretrain_cfg.get("checkpoint")
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        gen_state = _extract_generator_state(state_dict)
        generator.load_state_dict(gen_state)
        if is_main_process():
            print(f"Loaded pretrained generator from: {ckpt_path}")
    else:
        init_net(generator)
        if is_main_process():
            print("No pretrained checkpoint, training CUT finetune from scratch")

    if use_dist:
        generator = DDP(generator, device_ids=[device])

    # Losses
    criterion_l1 = nn.L1Loss()
    criterion_lpips = PerceptualLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)

    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=train_cfg.get("lr", 5e-5),
        betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)),
    )

    # Paired dataset
    modality = data_cfg["modalities"][0]
    crop_size_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    split = train_cfg.get("split") or data_cfg.get("split", "pro_train")

    if preprocessed_dir:
        dataset = CachedPairedDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modality=modality, source_field=data_cfg["source_field"],
            target_field=data_cfg["target_field"], crop_size=crop_size,
        )
    else:
        split_full = SPLIT_MAP.get(split, split)
        dataset = PairedMRIDataset(
            data_root=data_cfg["data_dir"], split=split_full,
            modality=modality, source_field=data_cfg["source_field"],
            target_field=data_cfg["target_field"], crop_size=crop_size,
        )

    batch_size = train_cfg.get("batch_size", 4)
    if use_dist:
        batch_size = batch_size // world_size
    sampler = DistributedSampler(dataset, shuffle=True) if use_dist else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    epochs = train_cfg.get("epochs", 50)
    lambda_l1 = finetune_cfg.get("loss_l1", 1.0)
    lambda_lpips = finetune_cfg.get("loss_lpips", 0.1)
    lambda_ssim = finetune_cfg.get("loss_ssim", 0.0)
    lambda_edge = finetune_cfg.get("loss_edge", 0.0)

    if lambda_edge > 0:
        criterion_edge = StructureLoss(ssim_weight=0.0, edge_weight=1.0).to(device)

    max_iters = train_cfg.get("max_iters_per_epoch", 0) or finetune_cfg.get("max_iters_per_epoch", 0)
    if is_main_process():
        print(f"CUT fine-tuning: {data_cfg['source_field']} -> {data_cfg['target_field']}, "
              f"{epochs} epochs, {len(loader)} iters/epoch"
              + (f" (capped at {max_iters})" if max_iters else "")
              + (f" (DDP: {world_size} GPUs)" if use_dist else ""))

    for epoch in range(epochs):
        generator.train()
        if use_dist and sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}",
                    disable=not is_main_process())

        for i, batch in enumerate(pbar):
            if max_iters and i >= max_iters:
                break
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            pred = generator(source)
            loss = lambda_l1 * criterion_l1(pred, target)
            loss += lambda_lpips * criterion_lpips(pred, target)
            if lambda_ssim > 0:
                loss += lambda_ssim * criterion_ssim(pred, target)
            if lambda_edge > 0:
                loss += lambda_edge * criterion_edge(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if is_main_process():
            avg = epoch_loss / max(len(loader), 1)
            print(f"Epoch {epoch+1}: avg_loss={avg:.6f}")

        if is_main_process() and (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            ckpt = weights_dir / f"checkpoint_epoch{epoch+1}.pth"
            net_G = generator.module if use_dist else generator
            torch.save({"epoch": epoch + 1, "generator": net_G.state_dict()}, ckpt)

    if is_main_process():
        net_G = generator.module if use_dist else generator
        torch.save(net_G.state_dict(), weights_dir / "generator_final.pth")
        print(f"CUT fine-tuning complete. Saved to: {output_dir}")


# --------------------------------------------------------------------------- #
#  CycleGAN Training
# --------------------------------------------------------------------------- #

def train_cyclegan(cfg: dict, device: torch.device, use_dist: bool = False):
    """Train CycleGAN model for a specific field-strength pair."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    rank, world_size = get_dist_info()
    output_dir = Path(train_cfg["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    model = CycleGANModel(
        input_nc=model_cfg.get("input_nc", 1),
        output_nc=model_cfg.get("output_nc", 1),
        ngf=model_cfg.get("ngf", 64),
        ndf=model_cfg.get("ndf", 64),
        n_blocks=model_cfg.get("n_blocks", 9),
        lambda_cycle=model_cfg.get("lambda_cycle", 10.0),
        lambda_idt=model_cfg.get("lambda_idt", 0.5),
        pool_size=model_cfg.get("pool_size", 50),
        gan_mode=model_cfg.get("gan_mode", "lsgan"),
        lr=train_cfg.get("lr", 0.0002),
        beta1=train_cfg.get("beta1", 0.5),
    ).to(device)
    model.init_weights()
    model.setup_optimizers()

    # DDP wrap sub-networks
    if use_dist:
        model.netG_AB = DDP(model.netG_AB, device_ids=[device])
        model.netG_BA = DDP(model.netG_BA, device_ids=[device])
        model.netD_A = DDP(model.netD_A, device_ids=[device])
        model.netD_B = DDP(model.netD_B, device_ids=[device])

    modality = data_cfg["modalities"][0]
    domain_a = data_cfg.get("domain_a") or data_cfg.get("source_field")
    domain_b = data_cfg.get("domain_b") or data_cfg.get("target_field")
    crop_size_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    split = train_cfg.get("split", "retro_train")

    if preprocessed_dir:
        dataset_a = CachedUnpairedDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modality=modality, field_strength=domain_a, crop_size=crop_size,
        )
        dataset_b = CachedUnpairedDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modality=modality, field_strength=domain_b, crop_size=crop_size,
        )
    else:
        split_full = SPLIT_MAP.get(split, split)
        dataset_a = UnpairedMRIDataset(
            data_root=data_cfg["data_dir"], split=split_full,
            modality=modality, field_strength=domain_a, crop_size=crop_size,
        )
        dataset_b = UnpairedMRIDataset(
            data_root=data_cfg["data_dir"], split=split_full,
            modality=modality, field_strength=domain_b, crop_size=crop_size,
        )

    batch_size = train_cfg.get("batch_size", 4)
    if use_dist:
        batch_size = batch_size // world_size
    num_workers = train_cfg.get("num_workers", 4)

    result = _make_unpaired_loaders(dataset_a, dataset_b, batch_size, num_workers, use_dist)
    if use_dist:
        loader_src, loader_tgt, sampler_src, sampler_tgt = result
        loader_len = min(len(loader_src), len(loader_tgt))
    else:
        loader = result[0]
        loader_len = len(loader)

    n_epochs = train_cfg.get("n_epochs", 100)
    n_epochs_decay = train_cfg.get("n_epochs_decay", 100)
    lr_lambda = get_lr_lambda(n_epochs, n_epochs_decay)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(model.optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(model.optimizer_D, lr_lambda)

    total_epochs = n_epochs + n_epochs_decay
    max_iters = train_cfg.get("max_iters_per_epoch", 0)
    if is_main_process():
        print(f"Training CycleGAN: {domain_a} <-> {domain_b}, "
              f"{total_epochs} epochs, {loader_len} iters/epoch"
              + (f" (capped at {max_iters})" if max_iters else "")
              + (f" (DDP: {world_size} GPUs)" if use_dist else ""))

    for epoch in range(total_epochs):
        model.train()
        if use_dist:
            sampler_src.set_epoch(epoch)
            sampler_tgt.set_epoch(epoch)
            iterable = _iter_unpaired((loader_src, loader_tgt), True)
        else:
            iterable = loader

        pbar = tqdm(iterable, desc=f"Epoch {epoch+1}/{total_epochs}",
                    total=loader_len, disable=not is_main_process())
        for i, (batch_a, batch_b) in enumerate(pbar):
            if max_iters and i >= max_iters:
                break
            real_A = batch_a["image"].to(device)
            real_B = batch_b["image"].to(device)
            losses = model.optimize_parameters(real_A, real_B)
            pbar.set_postfix(G=f"{losses['loss_G']:.4f}", D=f"{losses['loss_D']:.4f}")

        scheduler_G.step()
        scheduler_D.step()

        if is_main_process() and (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            ckpt = weights_dir / f"checkpoint_epoch{epoch+1}.pth"
            torch.save({"epoch": epoch + 1, "model": model.state_dict()}, ckpt)
            print(f"Saved: {ckpt}")

    if is_main_process():
        net_G_AB = model.netG_AB.module if use_dist else model.netG_AB
        net_G_BA = model.netG_BA.module if use_dist else model.netG_BA
        torch.save(net_G_AB.state_dict(), weights_dir / "generator_AB_final.pth")
        torch.save(net_G_BA.state_dict(), weights_dir / "generator_BA_final.pth")
        torch.save(net_G_AB.state_dict(), weights_dir / "generator_final.pth")
        print(f"CycleGAN training complete. Saved to: {output_dir}")


# --------------------------------------------------------------------------- #
#  CycleGAN Fine-tuning (paired, supervised)
# --------------------------------------------------------------------------- #

def train_cyclegan_finetune(cfg: dict, device: torch.device, use_dist: bool = False):
    """Phase 2 for CycleGAN: fine-tune the pretrained ResnetGenerator with paired data.

    Discards CycleGAN-specific structure (cycle loss, identity loss, dual G/D)
    and keeps only the source→target generator (G_AB). Supervised with paired
    (source, target) images using L1 + LPIPS (+ optional SSIM, Edge) loss.

    Used by --mode pro_pretrained (load CycleGAN G_AB pretrain ckpt) and
    --mode pro_scratch (random init).
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    pretrain_cfg = cfg.get("pretrain", {})
    finetune_cfg = cfg.get("finetune", {})
    rank, world_size = get_dist_info()
    output_dir = Path(train_cfg["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    # Build generator (same ResnetGenerator as CycleGAN G_AB)
    generator = ResnetGenerator(
        input_nc=model_cfg.get("input_nc", 1),
        output_nc=model_cfg.get("output_nc", 1),
        ngf=model_cfg.get("ngf", 64),
        n_blocks=model_cfg.get("n_blocks", 9),
    ).to(device)

    # Load pretrained weights
    ckpt_path = pretrain_cfg.get("checkpoint")
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        gen_state = _extract_generator_state(state_dict)
        generator.load_state_dict(gen_state)
        if is_main_process():
            print(f"Loaded pretrained generator from: {ckpt_path}")
    else:
        init_net(generator)
        if is_main_process():
            print("No pretrained checkpoint, training CycleGAN finetune from scratch")

    if use_dist:
        generator = DDP(generator, device_ids=[device])

    # Losses
    criterion_l1 = nn.L1Loss()
    criterion_lpips = PerceptualLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)

    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=train_cfg.get("lr", 5e-5),
        betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)),
    )

    # Paired dataset
    modality = data_cfg["modalities"][0]
    crop_size_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    split = train_cfg.get("split") or data_cfg.get("split", "pro_train")

    if preprocessed_dir:
        dataset = CachedPairedDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modality=modality, source_field=data_cfg["source_field"],
            target_field=data_cfg["target_field"], crop_size=crop_size,
        )
    else:
        split_full = SPLIT_MAP.get(split, split)
        dataset = PairedMRIDataset(
            data_root=data_cfg["data_dir"], split=split_full,
            modality=modality, source_field=data_cfg["source_field"],
            target_field=data_cfg["target_field"], crop_size=crop_size,
        )

    batch_size = train_cfg.get("batch_size", 4)
    if use_dist:
        batch_size = batch_size // world_size
    sampler = DistributedSampler(dataset, shuffle=True) if use_dist else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    epochs = train_cfg.get("epochs", 50)
    lambda_l1 = finetune_cfg.get("loss_l1", 1.0)
    lambda_lpips = finetune_cfg.get("loss_lpips", 0.1)
    lambda_ssim = finetune_cfg.get("loss_ssim", 0.0)
    lambda_edge = finetune_cfg.get("loss_edge", 0.0)

    if lambda_edge > 0:
        criterion_edge = StructureLoss(ssim_weight=0.0, edge_weight=1.0).to(device)

    max_iters = train_cfg.get("max_iters_per_epoch", 0) or finetune_cfg.get("max_iters_per_epoch", 0)
    if is_main_process():
        print(f"CycleGAN fine-tuning: {data_cfg['source_field']} -> {data_cfg['target_field']}, "
              f"{epochs} epochs, {len(loader)} iters/epoch"
              + (f" (capped at {max_iters})" if max_iters else "")
              + (f" (DDP: {world_size} GPUs)" if use_dist else ""))

    for epoch in range(epochs):
        generator.train()
        if use_dist and sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}",
                    disable=not is_main_process())

        for i, batch in enumerate(pbar):
            if max_iters and i >= max_iters:
                break
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            pred = generator(source)
            loss = lambda_l1 * criterion_l1(pred, target)
            loss += lambda_lpips * criterion_lpips(pred, target)
            if lambda_ssim > 0:
                loss += lambda_ssim * criterion_ssim(pred, target)
            if lambda_edge > 0:
                loss += lambda_edge * criterion_edge(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if is_main_process():
            avg = epoch_loss / max(len(loader), 1)
            print(f"Epoch {epoch+1}: avg_loss={avg:.6f}")

        if is_main_process() and (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            ckpt = weights_dir / f"checkpoint_epoch{epoch+1}.pth"
            net_G = generator.module if use_dist else generator
            torch.save({"epoch": epoch + 1, "generator": net_G.state_dict()}, ckpt)

    if is_main_process():
        net_G = generator.module if use_dist else generator
        torch.save(net_G.state_dict(), weights_dir / "generator_final.pth")
        print(f"CycleGAN fine-tuning complete. Saved to: {output_dir}")


# --------------------------------------------------------------------------- #
#  StarGAN v2 Training
# --------------------------------------------------------------------------- #

def train_stargan(cfg: dict, device: torch.device, use_dist: bool = False,
                  resume_ckpt: str = None):
    """Train StarGAN v2 for multi-domain translation (Task 3)."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    domains = cfg["domains"]
    rank, world_size = get_dist_info()
    output_dir = Path(train_cfg["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    nets, nets_ema = build_stargan_v2(
        img_size=model_cfg.get("img_size", 128),
        style_dim=model_cfg.get("style_dim", 64),
        latent_dim=model_cfg.get("latent_dim", 16),
        num_domains=model_cfg.get("num_domains", 5),
        max_conv_dim=model_cfg.get("max_conv_dim", 512),
        input_nc=model_cfg.get("input_nc", 1),
    )
    for key in nets:
        nets[key] = nets[key].to(device)
    for key in nets_ema:
        nets_ema[key] = nets_ema[key].to(device)

    # DDP wrap training networks
    if use_dist:
        for key in nets:
            nets[key] = DDP(nets[key], device_ids=[device])

    modality = data_cfg["modalities"][0]
    crop_size_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    split = train_cfg.get("split", "retro_train")

    if preprocessed_dir:
        dataset = CachedMultiDomainDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modality=modality, field_strengths=domains, crop_size=crop_size,
        )
    else:
        split_full = SPLIT_MAP.get(split, split)
        dataset = MultiDomainMRIDataset(
            data_root=data_cfg["data_dir"], split=split_full,
            modality=modality, field_strengths=domains, crop_size=crop_size,
        )

    batch_size = train_cfg.get("batch_size", 4)
    if use_dist:
        batch_size = batch_size // world_size
    sampler = DistributedSampler(dataset, shuffle=True) if use_dist else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
    )

    # Optimizers (use .module params if DDP-wrapped)
    def _params(key):
        return nets[key].module.parameters() if use_dist else nets[key].parameters()

    lr = train_cfg.get("lr", 0.0001)
    f_lr = train_cfg.get("f_lr", 1e-6)
    beta1 = train_cfg.get("beta1", 0.0)
    beta2 = train_cfg.get("beta2", 0.99)
    wd = train_cfg.get("weight_decay", 1e-4)

    optims = {
        "generator": torch.optim.Adam(_params("generator"),
                                       lr=lr, betas=(beta1, beta2), weight_decay=wd),
        "mapping_network": torch.optim.Adam(_params("mapping_network"),
                                             lr=f_lr, betas=(beta1, beta2), weight_decay=wd),
        "style_encoder": torch.optim.Adam(_params("style_encoder"),
                                           lr=lr, betas=(beta1, beta2), weight_decay=wd),
        "discriminator": torch.optim.Adam(_params("discriminator"),
                                           lr=lr, betas=(beta1, beta2), weight_decay=wd),
    }

    latent_dim = model_cfg.get("latent_dim", 16)
    total_iters = train_cfg.get("total_iters", 200000)
    lambda_ds = train_cfg.get("lambda_ds", 1.0)
    ds_iter = train_cfg.get("ds_iter", 100000)
    ema_beta = train_cfg.get("ema_beta", 0.999)

    # For compute_d_loss / compute_g_loss, pass unwrapped nets
    def _unwrap(d):
        return {k: (v.module if use_dist else v) for k, v in d.items()}

    # Resume from checkpoint
    step = 0
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        nets_unwrap = _unwrap(nets)
        for k in nets_unwrap:
            nets_unwrap[k].load_state_dict(ckpt["nets"][k])
        for k in nets_ema:
            nets_ema[k].load_state_dict(ckpt["nets_ema"][k])
        if "optims" in ckpt:
            for k in optims:
                if k in ckpt["optims"]:
                    optims[k].load_state_dict(ckpt["optims"][k])
        step = ckpt.get("step", 0)
        if is_main_process():
            print(f"Resumed from {resume_ckpt} at step {step}")

    # LR scheduler: linear decay in the second half of training
    lr_decay_start = train_cfg.get("lr_decay_start", total_iters // 2)
    def _lr_lambda(current_step):
        if current_step < lr_decay_start:
            return 1.0
        return 1.0 - (current_step - lr_decay_start) / (total_iters - lr_decay_start + 1)

    schedulers = {
        k: torch.optim.lr_scheduler.LambdaLR(v, _lr_lambda, last_epoch=step - 1 if step > 0 else -1)
        for k, v in optims.items()
    }

    if is_main_process():
        print(f"Training StarGAN v2: {len(domains)} domains, {total_iters} iters"
              + (f" (DDP: {world_size} GPUs)" if use_dist else "")
              + (f" (resuming from step {step})" if step > 0 else "")
              + f" (LR decay from step {lr_decay_start})")
    epoch_count = 0
    while step < total_iters:
        if use_dist and sampler is not None:
            sampler.set_epoch(epoch_count)
        epoch_count += 1
        for batch in loader:
            if step >= total_iters:
                break

            x_real = batch["image"].to(device)
            y_org = batch["domain"].to(device).long()
            x_ref = batch["ref_image"].to(device)
            y_trg = batch["ref_domain"].to(device).long()

            z_trg = torch.randn(x_real.size(0), latent_dim, device=device)
            z_trg2 = torch.randn(x_real.size(0), latent_dim, device=device)

            current_lambda_ds = lambda_ds if ds_iter <= 0 else \
                lambda_ds * max(0, 1 - step / ds_iter)

            # Alternate between latent-guided and reference-guided
            use_latent = (step % 2 == 0)

            # D step
            if use_latent:
                d_loss, d_losses = compute_d_loss(
                    _unwrap(nets), x_real, y_org, y_trg, z_trg=z_trg,
                    lambda_reg=train_cfg.get("lambda_reg", 1.0),
                )
            else:
                d_loss, d_losses = compute_d_loss(
                    _unwrap(nets), x_real, y_org, y_trg, x_ref=x_ref,
                    lambda_reg=train_cfg.get("lambda_reg", 1.0),
                )
            _reset_grad(optims)
            d_loss.backward()
            optims["discriminator"].step()

            # G step
            if use_latent:
                g_loss, g_losses = compute_g_loss(
                    _unwrap(nets), x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2],
                    lambda_sty=train_cfg.get("lambda_sty", 1.0),
                    lambda_ds=current_lambda_ds,
                    lambda_cyc=train_cfg.get("lambda_cyc", 1.0),
                )
            else:
                # For reference-guided, use x_ref and a shuffled version as x_ref2
                perm = torch.randperm(x_ref.size(0), device=device)
                x_ref2 = x_ref[perm]
                g_loss, g_losses = compute_g_loss(
                    _unwrap(nets), x_real, y_org, y_trg, x_refs=[x_ref, x_ref2],
                    lambda_sty=train_cfg.get("lambda_sty", 1.0),
                    lambda_ds=current_lambda_ds,
                    lambda_cyc=train_cfg.get("lambda_cyc", 1.0),
                )
            _reset_grad(optims)
            g_loss.backward()
            optims["generator"].step()
            optims["mapping_network"].step()
            optims["style_encoder"].step()

            # EMA update (on unwrapped parameters)
            nets_unwrap = _unwrap(nets)
            moving_average(nets_unwrap["generator"], nets_ema["generator"], ema_beta)
            moving_average(nets_unwrap["mapping_network"], nets_ema["mapping_network"], ema_beta)
            moving_average(nets_unwrap["style_encoder"], nets_ema["style_encoder"], ema_beta)

            # LR scheduler step
            for sched in schedulers.values():
                sched.step()

            step += 1
            if is_main_process() and step % train_cfg.get("print_every", 1000) == 0:
                current_lr = schedulers["generator"].get_last_lr()[0]
                print(f"[{step}/{total_iters}] D: {d_loss.item():.4f}, "
                      f"G: {g_loss.item():.4f}, ds: {current_lambda_ds:.3f}, lr: {current_lr:.2e}")

            if is_main_process() and step % train_cfg.get("save_every", 10000) == 0:
                ckpt = weights_dir / f"checkpoint_{step}.pth"
                torch.save({
                    "step": step,
                    "nets": {k: v.state_dict() for k, v in _unwrap(nets).items()},
                    "nets_ema": {k: v.state_dict() for k, v in nets_ema.items()},
                    "optims": {k: v.state_dict() for k, v in optims.items()},
                }, ckpt)
                print(f"Saved: {ckpt}")

    if is_main_process():
        torch.save({
            "nets": {k: v.state_dict() for k, v in _unwrap(nets).items()},
            "nets_ema": {k: v.state_dict() for k, v in nets_ema.items()},
        }, weights_dir / "model_final.pth")
        print(f"StarGAN v2 training complete. Saved to: {output_dir}")


def train_stargan_finetune(cfg: dict, device: torch.device, use_dist: bool = False):
    """Fine-tune pretrained StarGAN v2 with paired data (L1 + LPIPS).

    Uses prospective paired data to supervise the multi-domain generator.
    For each batch, samples paired (source, target) images across all domain
    combinations, translates source→target, and computes reconstruction loss.
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    domains = cfg["domains"]
    finetune_cfg = cfg.get("finetune", {})
    rank, world_size = get_dist_info()
    output_dir = Path(train_cfg["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    # Build StarGAN v2 networks
    nets, nets_ema = build_stargan_v2(
        img_size=model_cfg.get("img_size", 128),
        style_dim=model_cfg.get("style_dim", 64),
        latent_dim=model_cfg.get("latent_dim", 16),
        num_domains=model_cfg.get("num_domains", 5),
        max_conv_dim=model_cfg.get("max_conv_dim", 512),
        input_nc=model_cfg.get("input_nc", 1),
    )

    # Load pretrained weights
    pretrain_ckpt = cfg.get("pretrain", {}).get("checkpoint")
    if pretrain_ckpt:
        state_dict = torch.load(pretrain_ckpt, map_location=device, weights_only=True)
        if "nets_ema" in state_dict:
            for k in nets_ema:
                nets_ema[k].load_state_dict(state_dict["nets_ema"][k])
            # Initialize training nets from EMA (better starting point)
            for k in nets_ema:
                if k in nets:
                    nets[k].load_state_dict(state_dict["nets_ema"][k])
        elif "nets" in state_dict:
            for k in nets:
                nets[k].load_state_dict(state_dict["nets"][k])
        if is_main_process():
            print(f"Loaded pretrained StarGAN v2 from: {pretrain_ckpt}")
    else:
        if is_main_process():
            print("No pretrained checkpoint, training StarGAN v2 finetune from scratch")

    for key in nets:
        nets[key] = nets[key].to(device)
    for key in nets_ema:
        nets_ema[key] = nets_ema[key].to(device)

    if use_dist:
        for key in nets:
            nets[key] = DDP(nets[key], device_ids=[device])

    # Build paired datasets for all domain combinations
    from mrixfields.data.utils import FIELD_TO_DOMAIN
    modality = data_cfg["modalities"][0]
    crop_size_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    split = train_cfg.get("split") or finetune_cfg.get("split", "pro_train")

    all_pairs = []
    for src_fs in domains:
        for tgt_fs in domains:
            if src_fs == tgt_fs:
                continue
            try:
                if preprocessed_dir:
                    ds = CachedPairedDataset(
                        preprocessed_dir=preprocessed_dir, split=split,
                        modality=modality, source_field=src_fs,
                        target_field=tgt_fs, crop_size=crop_size,
                    )
                else:
                    split_full = SPLIT_MAP.get(split, split)
                    ds = PairedMRIDataset(
                        data_root=data_cfg["data_dir"], split=split_full,
                        modality=modality, source_field=src_fs,
                        target_field=tgt_fs, crop_size=crop_size,
                    )
                all_pairs.append((src_fs, tgt_fs, ds))
            except (FileNotFoundError, ValueError):
                continue

    if not all_pairs:
        raise FileNotFoundError("No paired data found for any domain combination")

    # Combine all pairs into one dataset with domain labels
    from torch.utils.data import ConcatDataset

    class _PairedWithDomains(torch.utils.data.Dataset):
        """Wraps a PairedMRIDataset to add domain indices."""
        def __init__(self, dataset, src_domain, tgt_domain):
            self.dataset = dataset
            self.src_domain = src_domain
            self.tgt_domain = tgt_domain
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            batch = self.dataset[idx]
            batch["src_domain"] = self.src_domain
            batch["tgt_domain"] = self.tgt_domain
            return batch

    wrapped = []
    for src_fs, tgt_fs, ds in all_pairs:
        wrapped.append(_PairedWithDomains(ds, FIELD_TO_DOMAIN[src_fs], FIELD_TO_DOMAIN[tgt_fs]))
        if is_main_process():
            print(f"  Paired data: {src_fs}→{tgt_fs}: {len(ds)} samples")
    combined_dataset = ConcatDataset(wrapped)

    batch_size = train_cfg.get("batch_size", 4)
    if use_dist:
        batch_size = batch_size // world_size
    sampler = DistributedSampler(combined_dataset, shuffle=True) if use_dist else None

    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
    )

    # Optimizer: only fine-tune generator and mapping network
    def _params(key):
        return nets[key].module.parameters() if use_dist else nets[key].parameters()

    lr = train_cfg.get("lr", 5e-5)
    optimizer = torch.optim.Adam(
        itertools.chain(_params("generator"), _params("mapping_network")),
        lr=lr,
        betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)),
    )

    # Losses
    criterion_l1 = nn.L1Loss()
    criterion_lpips = PerceptualLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)

    lambda_l1 = finetune_cfg.get("loss_l1", 1.0)
    lambda_lpips = finetune_cfg.get("loss_lpips", 0.1)
    lambda_ssim = finetune_cfg.get("loss_ssim", 0.0)
    lambda_edge = finetune_cfg.get("loss_edge", 0.0)
    if lambda_edge > 0:
        criterion_edge = StructureLoss(ssim_weight=0.0, edge_weight=1.0).to(device)

    epochs = train_cfg.get("epochs", 50)
    latent_dim = model_cfg.get("latent_dim", 16)
    ema_beta = train_cfg.get("ema_beta", finetune_cfg.get("ema_beta", 0.999))

    def _unwrap(d):
        return {k: (v.module if use_dist else v) for k, v in d.items()}

    if is_main_process():
        print(f"StarGAN v2 fine-tuning: {len(all_pairs)} pairs, {len(combined_dataset)} samples, "
              f"{epochs} epochs" + (f" (DDP: {world_size} GPUs)" if use_dist else ""))

    for epoch in range(epochs):
        if use_dist and sampler is not None:
            sampler.set_epoch(epoch)
        nets["generator"].train()
        nets["mapping_network"].train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") if is_main_process() else loader

        for batch in pbar:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            tgt_domain = batch["tgt_domain"].to(device).long()

            # Generate style code for target domain
            z = torch.randn(source.size(0), latent_dim, device=device)
            nets_unwrap = _unwrap(nets)
            s_trg = nets_unwrap["mapping_network"](z, tgt_domain)
            pred = nets_unwrap["generator"](source, s_trg)

            # Reconstruction losses
            loss = lambda_l1 * criterion_l1(pred, target)
            loss = loss + lambda_lpips * criterion_lpips(pred, target)
            if lambda_ssim > 0:
                loss = loss + lambda_ssim * criterion_ssim(pred, target)
            if lambda_edge > 0:
                loss = loss + lambda_edge * criterion_edge(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            moving_average(nets_unwrap["generator"], nets_ema["generator"], ema_beta)
            moving_average(nets_unwrap["mapping_network"], nets_ema["mapping_network"], ema_beta)

            epoch_loss += loss.item()
            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if is_main_process():
            avg = epoch_loss / max(len(loader), 1)
            print(f"Epoch {epoch+1}: avg_loss={avg:.6f}")

        if is_main_process() and (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            ckpt = weights_dir / f"checkpoint_epoch{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "nets": {k: v.state_dict() for k, v in _unwrap(nets).items()},
                "nets_ema": {k: v.state_dict() for k, v in nets_ema.items()},
            }, ckpt)

    if is_main_process():
        torch.save({
            "nets": {k: v.state_dict() for k, v in _unwrap(nets).items()},
            "nets_ema": {k: v.state_dict() for k, v in nets_ema.items()},
        }, weights_dir / "model_final.pth")
        print(f"StarGAN v2 fine-tuning complete. Saved to: {output_dir}")


def _reset_grad(optims):
    for optim in optims.values():
        optim.zero_grad()


def _extract_generator_state(state_dict: dict) -> dict:
    """Extract generator weights from various checkpoint formats.

    Supports:
        - Full CUT/CycleGAN model checkpoint (key "model" with "netG." prefix)
        - Hybrid fine-tune checkpoint (key "generator")
        - Raw generator state_dict (no wrapper keys)

    Raises:
        ValueError: If no generator weights can be found.
    """
    # Format 1: CUT/CycleGAN full model checkpoint {"model": {"netG.xxx": ...}}
    if "model" in state_dict:
        prefix = "netG."
        gen_state = {k[len(prefix):]: v for k, v in state_dict["model"].items()
                     if k.startswith(prefix)}
        if gen_state:
            return gen_state
        # CycleGAN AB direction
        prefix_ab = "netG_AB."
        gen_state = {k[len(prefix_ab):]: v for k, v in state_dict["model"].items()
                     if k.startswith(prefix_ab)}
        if gen_state:
            return gen_state
        raise ValueError(
            f"Checkpoint has 'model' key but no generator prefix found. "
            f"Available prefixes: {set(k.split('.')[0] for k in state_dict['model'])}"
        )

    # Format 2: Hybrid fine-tune checkpoint {"generator": {...}}
    if "generator" in state_dict:
        return state_dict["generator"]

    # Format 3: Raw state_dict (direct generator weights)
    if all(not k.startswith("net") for k in state_dict):
        return state_dict

    raise ValueError(
        f"Unrecognized checkpoint format. Top-level keys: {list(state_dict.keys())}"
    )


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def _save_actual_config(cfg: dict, output_dir: Path):
    """Save the merged runtime config (yaml + CLI overrides) for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_actual.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _build_pretrain_cfg(cfg: dict, output_dir: str) -> dict:
    """Build a config dict for the pretrain step by merging top-level and pretrain sections."""
    pretrain_cfg = cfg["pretrain"]
    return {
        **cfg,
        "training": {
            **pretrain_cfg,
            "output_dir": output_dir,
        },
    }


def _build_stargan_finetune_cfg(cfg: dict, output_dir: str, pretrain_ckpt: str) -> dict:
    """Build a config dict for StarGAN v2 fine-tuning (keeps multi-domain architecture)."""
    ft_cfg = cfg["finetune"]
    return {
        **cfg,
        "pretrain": {"method": "stargan_v2", "checkpoint": pretrain_ckpt},
        "training": {
            **ft_cfg,
            "output_dir": output_dir,
        },
    }


def _build_finetune_cfg(cfg: dict, output_dir: str, pretrain_ckpt: str) -> dict:
    """Build a config dict for the finetune step (CUT/CycleGAN paired finetune).

    Keeps the original method ("cut" or "cyclegan") in the cfg so the main
    dispatcher can route to train_cut_finetune / train_cyclegan_finetune.
    """
    ft_cfg = cfg["finetune"]
    return {
        **cfg,
        "pretrain": {"method": cfg["method"], "checkpoint": pretrain_ckpt},
        "model": {
            "name": "resnet",
            "input_nc": cfg["model"].get("input_nc", 1),
            "output_nc": cfg["model"].get("output_nc", 1),
            "ngf": cfg["model"].get("ngf", 64),
            "n_blocks": cfg["model"].get("n_blocks", 9),
        },
        "training": {
            **ft_cfg,
            "output_dir": output_dir,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="MRIxFields2026 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="retro_scratch",
                        choices=["retro_scratch", "pro_scratch", "pro_pretrained"],
                        help="Training mode: retro_scratch (pretrain only), pro_scratch (supervised from scratch), pro_pretrained (finetune pretrained model)")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Path to pretrained generator checkpoint (skip pretrain step)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training (StarGAN v2 only)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dist", action="store_true",
                        help="Enable distributed training (use with torchrun)")
    args = parser.parse_args()

    # Initialize distributed training
    use_dist = args.dist
    if use_dist:
        local_rank = init_dist()
        device = torch.device(f"cuda:{local_rank}")
    else:
        cfg_tmp = load_config(args.config)  # peek for device
        load_env()
        env_device = get_device()
        device = torch.device(args.device if args.device != "cuda" else env_device)
        if not torch.cuda.is_available():
            device = torch.device("cpu")

    cfg = load_config(args.config)

    # Inject paths from .env into config
    load_env()
    cfg.setdefault("data", {})
    cfg["data"]["data_dir"] = get_data_dir()
    cfg["data"]["preprocessed_dir"] = get_preprocessed_dir()

    method = cfg["method"]
    task_name = cfg.get("task_name", f"{method}_unknown")
    rank, world_size = get_dist_info()

    # Record runtime info
    cfg["_runtime"] = {
        "mode": args.mode,
        "config_path": str(Path(args.config).resolve()),
        "pretrain_ckpt": args.pretrain_ckpt,
        "device": str(device),
        "dist": use_dist,
        "world_size": world_size,
    }

    # Output dirs with mode suffix
    output_dir = get_output_dir()
    output_base = Path(output_dir) / task_name / method / args.mode

    if is_main_process():
        print(f"Device: {device}")
        print(f"Task:   {task_name}")
        print(f"Mode:   {args.mode}")
        if use_dist:
            print(f"DDP:    {world_size} GPUs")
        print(f"Output: {output_base}")
        _save_actual_config(cfg, output_base)

    # ------------------------------------------------------------------ #
    #  Mode: retro_scratch — pretrain only (unpaired, retrospective data)
    # ------------------------------------------------------------------ #
    if args.mode == "retro_scratch":
        output_dir = str(output_base)
        pretrain_run_cfg = _build_pretrain_cfg(cfg, output_dir)

        if method == "cut":
            train_cut(pretrain_run_cfg, device, use_dist)
        elif method == "cyclegan":
            train_cyclegan(pretrain_run_cfg, device, use_dist)
        elif method == "stargan_v2":
            train_stargan(pretrain_run_cfg, device, use_dist, resume_ckpt=args.resume)
        else:
            raise ValueError(f"Unknown method for pretrain: {method}")

    # ------------------------------------------------------------------ #
    #  Mode: pro_pretrained — finetune a pretrained model (paired, prospective)
    # ------------------------------------------------------------------ #
    elif args.mode == "pro_pretrained":
        pretrain_ckpt = args.pretrain_ckpt
        if pretrain_ckpt is None:
            if method == "stargan_v2":
                default_ckpt = output_base.parent / "retro_scratch" / "weights" / "model_final.pth"
            else:
                default_ckpt = output_base.parent / "retro_scratch" / "weights" / "generator_final.pth"
            if not default_ckpt.exists():
                raise FileNotFoundError(
                    f"No pretrain checkpoint found at {default_ckpt}. "
                    f"Run --mode retro_scratch first, or use --pretrain_ckpt to specify."
                )
            pretrain_ckpt = str(default_ckpt)
        if is_main_process():
            print(f"Loading pretrain checkpoint: {pretrain_ckpt}")

        if method == "stargan_v2":
            ft_run_cfg = _build_stargan_finetune_cfg(cfg, str(output_base), pretrain_ckpt)
            train_stargan_finetune(ft_run_cfg, device, use_dist)
        elif method == "cut":
            ft_run_cfg = _build_finetune_cfg(cfg, str(output_base), pretrain_ckpt)
            train_cut_finetune(ft_run_cfg, device, use_dist)
        elif method == "cyclegan":
            ft_run_cfg = _build_finetune_cfg(cfg, str(output_base), pretrain_ckpt)
            train_cyclegan_finetune(ft_run_cfg, device, use_dist)
        else:
            raise ValueError(f"Unknown method for pro_pretrained: {method}")

    # ------------------------------------------------------------------ #
    #  Mode: pro_scratch — supervised training from scratch (paired, prospective)
    # ------------------------------------------------------------------ #
    elif args.mode == "pro_scratch":
        if is_main_process():
            print("Mode=pro_scratch: training from random initialization (no pretrain checkpoint)")
        if method == "stargan_v2":
            ft_run_cfg = _build_stargan_finetune_cfg(cfg, str(output_base), None)
            train_stargan_finetune(ft_run_cfg, device, use_dist)
        elif method == "cut":
            ft_run_cfg = _build_finetune_cfg(cfg, str(output_base), None)
            train_cut_finetune(ft_run_cfg, device, use_dist)
        elif method == "cyclegan":
            ft_run_cfg = _build_finetune_cfg(cfg, str(output_base), None)
            train_cyclegan_finetune(ft_run_cfg, device, use_dist)
        else:
            raise ValueError(f"Unknown method for pro_scratch: {method}")

    if is_main_process():
        print(f"\nTraining complete ({args.mode}). Output: {output_base}")

    if use_dist:
        cleanup_dist()


if __name__ == "__main__":
    main()
