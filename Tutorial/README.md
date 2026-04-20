# MRIxFields2026 — Tutorial

Step-by-step Jupyter notebooks for getting started with the MRIxFields2026 challenge.

## Prerequisites

Ensure you have set up the environment. See [main README](../README.md#3-set-up-environment).

- Data placed in `$DATA_DIR` (configured in `.env`)
- SynthSeg installed at `$SYNTHSEG_DIR` (required by notebook 03, see [Evaluation/README.md](../Evaluation/README.md))

## Getting Started

```bash
conda activate mf
cd Tutorial
jupyter notebook
```

## Notebooks

Notebooks are designed to run sequentially — each one builds on the outputs of the previous.

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [01_data_exploration.ipynb](01_data_exploration.ipynb) | Load NIfTI volumes, inspect headers/shapes (364×436×364), visualize across 5 field strengths and 3 modalities |
| 02 | [02_baseline.ipynb](02_baseline.ipynb) | Self-contained CUT baseline (0.1T → 7T): preprocessing → model definition → pretrain (5 epochs) → finetune (2 epochs) → inference on subject 0007 → saves prediction to `tmp/inference/` |
| 03 | [03_evaluation.ipynb](03_evaluation.ipynb) | Evaluate predictions from notebook 02: voxel metrics (nRMSE, SSIM, LPIPS), error map visualization, SynthSeg segmentation, DGM overlay visualization, Dice and volume consistency per structure |

## Intermediate Files

Notebooks 02 and 03 write intermediate results to `tmp/`:

```
tmp/
├── preprocessed/              # 2D axial slices (NPZ) from notebook 02
├── pretrain_generator.pth     # CUT pretrained checkpoint
├── finetune_generator.pth     # Finetuned checkpoint
├── inference/                 # Predicted + source NIfTI volumes
│   ├── P_T1W_0.1T_0007.nii.gz
│   └── P_T1W_7T_0007_pred.nii.gz
└── seg/                       # SynthSeg segmentations from notebook 03
    ├── P_T1W_7T_0007_pred_seg.nii.gz
    └── P_T1W_7T_0007_seg.nii.gz
```

For data split details, see the [main README](../README.md#2-get-the-data).
