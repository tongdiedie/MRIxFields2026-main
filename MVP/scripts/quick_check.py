#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp.networks import ConditionalUNet3D, FieldDegradationSimulator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 1, 16, 32, 32, device=device)
    g = ConditionalUNet3D(base_ch=4, levels=3, cond_dim=32, use_coords=True).to(device)
    fds = FieldDegradationSimulator(cond_dim=32, residual_ch=4, use_coords=True).to(device)
    with torch.no_grad():
        y = g(x, ["0.1T"], ["7T"], ["T1W"])
        z = fds(x, ["7T"], ["0.1T"], ["T1W"])
    print("device:", device)
    print("translator output:", tuple(y.shape), float(y.min()), float(y.max()))
    print("fds output:", tuple(z.shape), float(z.min()), float(z.max()))
    print("OK")


if __name__ == "__main__":
    main()
