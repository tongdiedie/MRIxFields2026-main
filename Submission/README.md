# Submission Guidelines

> **Note**: This document is subject to change. Please check back for updates.

## Validation Phase

The challenge has three independent tasks. Each task is a set of directed
`source field → target field` translations across 3 modalities (T1W / T2W /
T2FLAIR), and you submit **one zip per task** to the platform. The figure
below maps out every subtask — yellow boxes are the inputs you receive, red
boxes are the targets you generate, and each arrow is one subtask.

<div align="center">
  <img src="../assets/fig_task_setting.png" alt="Task 1/2/3 subtask diagram — yellow = input field, red = target field" width="700">
</div>

Submit via the [Synapse platform](https://www.synapse.org/Synapse:syn72060672):

| Zip filename | Task | Pairs | seg/ |
|--------------|------|-------|------|
| `task1.zip` | Any → 7T | 4 (`0.1T_to_7T`, `1.5T_to_7T`, `3T_to_7T`, `5T_to_7T`) | **mandatory** |
| `task2.zip` | 0.1T → Higher | 4 (`0.1T_to_1.5T`, `0.1T_to_3T`, `0.1T_to_5T`, `0.1T_to_7T`) | **mandatory** |
| `task3.zip` | Any → Any (unified model) | 20 directed pairs (5 src × 4 tgt, incl. downfield) | not used |

**Do not** bundle multiple tasks into one zip — each task is a separate Synapse upload. If you don't participate in a given task, just skip its zip.

### Subject IDs (validation phase)

The validation cohort is paired multi-field: every patient has scans at **all 5 field strengths × 3 modalities** in our private ground truth. The public input release exposes only **3 patient IDs per source field**, with disjoint ID sets per source field — every patient is given to participants in **exactly one** source field, and the same patient's other-field scans are held out as paired GT for evaluation.

| Source field | Released input IDs |
|---|---|
| `0.1T` | `0001`, `0002`, `0003` |
| `1.5T` | `0004`, `0005`, `0008` |
| `3T`   | `0010`, `0011`, `0012` |
| `5T`   | `0013`, `0014`, `0015` |
| `7T`   | `0016`, `0017`, `0018` |

**Naming rule**: keep the source ID and replace the field tag with the target field. For example, given source input `P_T1W_0.1T_0001.nii.gz`, the `0.1T_to_7T` prediction is `P_T1W_7T_0001.nii.gz`. The evaluator looks up patient `0001` in the private 7T GT for the comparison.

So expected file IDs per `(modality, pair)` follow the source field of that pair:

#### Task 1 — `Any → 7T` (12 subtasks; `pred/` + `seg/`)
| pair | IDs |
|---|---|
| `0.1T_to_7T` | `0001`, `0002`, `0003` |
| `1.5T_to_7T` | `0004`, `0005`, `0008` |
| `3T_to_7T`   | `0010`, `0011`, `0012` |
| `5T_to_7T`   | `0013`, `0014`, `0015` |

#### Task 2 — `0.1T → Higher` (12 subtasks; `pred/` + `seg/`)
All four pairs use the 0.1T set: **`0001`, `0002`, `0003`**.

#### Task 3 — `Any → Any` (60 subtasks; `pred/` only)
| Source field of pair | IDs |
|---|---|
| `0.1T_to_*` | `0001`, `0002`, `0003` |
| `1.5T_to_*` | `0004`, `0005`, `0008` |
| `3T_to_*`   | `0010`, `0011`, `0012` |
| `5T_to_*`   | `0013`, `0014`, `0015` |
| `7T_to_*`   | `0016`, `0017`, `0018` |

### Directory Structure (inside each zip)

The zip filename already encodes the task, so each zip's internal layout starts at the modality level — no `task{N}/` prefix. This mirrors the dataset's `{modality}/{field}/<file>.nii.gz` ordering one-to-one (the submission's `pair` plays the role of the dataset's `field`). Filenames are identical to the corresponding GT file.

```
task1.zip                                     # Task 1: Any -> 7T  (4 pairs × 3 modalities × 3 subjects)
└── (zip root)
    ├── T1W/
    │   ├── 0.1T_to_7T/                       # IDs 0001, 0002, 0003
    │   │   ├── pred/
    │   │   │   ├── P_T1W_7T_0001.nii.gz
    │   │   │   ├── P_T1W_7T_0002.nii.gz
    │   │   │   └── P_T1W_7T_0003.nii.gz
    │   │   └── seg/                          # mandatory for task1 — drives Dice / Volume
    │   │       ├── P_T1W_7T_0001_seg.nii.gz
    │   │       ├── P_T1W_7T_0002_seg.nii.gz
    │   │       └── P_T1W_7T_0003_seg.nii.gz
    │   ├── 1.5T_to_7T/                       # IDs 0004, 0005, 0008  (same pred/ + seg/ shape)
    │   ├── 3T_to_7T/                         # IDs 0010, 0011, 0012
    │   └── 5T_to_7T/                         # IDs 0013, 0014, 0015
    ├── T2W/                                  # same 4 pairs × {pred, seg} shape as T1W
    └── T2FLAIR/

task2.zip                                     # Task 2: 0.1T -> Higher  (4 pairs × 3 modalities × 3 subjects)
└── (zip root)
    ├── T1W/
    │   ├── 0.1T_to_1.5T/                     # all task2 pairs use IDs 0001, 0002, 0003
    │   ├── 0.1T_to_3T/
    │   ├── 0.1T_to_5T/
    │   └── 0.1T_to_7T/                       # each: pred/ (3 files) + seg/ (3 files)
    ├── T2W/
    └── T2FLAIR/

task3.zip                                     # Task 3: Any -> Any  (20 pairs × 3 modalities × 3 subjects, pred/ only)
└── (zip root)
    ├── T1W/                                  # 5 source fields × 4 target fields = 20 directed pairs (incl. downfield)
    │   ├── 0.1T_to_1.5T/
    │   │   └── pred/                         # task3 has no seg/ — any seg/ inside task3.zip is ignored
    │   │       ├── P_T1W_1.5T_0001.nii.gz    # IDs from source field (0.1T → 0001/0002/0003)
    │   │       ├── P_T1W_1.5T_0002.nii.gz
    │   │       └── P_T1W_1.5T_0003.nii.gz
    │   ├── 0.1T_to_3T/                       # IDs 0001, 0002, 0003
    │   ├── 0.1T_to_5T/
    │   ├── 0.1T_to_7T/
    │   ├── 1.5T_to_0.1T/                     # IDs 0004, 0005, 0008
    │   ├── 1.5T_to_3T/
    │   ├── 1.5T_to_5T/
    │   ├── 1.5T_to_7T/
    │   ├── 3T_to_0.1T/                       # IDs 0010, 0011, 0012
    │   ├── 3T_to_1.5T/
    │   ├── 3T_to_5T/
    │   ├── 3T_to_7T/
    │   ├── 5T_to_0.1T/                       # IDs 0013, 0014, 0015
    │   ├── 5T_to_1.5T/
    │   ├── 5T_to_3T/
    │   ├── 5T_to_7T/
    │   ├── 7T_to_0.1T/                       # IDs 0016, 0017, 0018
    │   ├── 7T_to_1.5T/
    │   ├── 7T_to_3T/
    │   └── 7T_to_5T/
    ├── T2W/
    └── T2FLAIR/
```

**Per-task contents** (validation phase: 3 input subjects per `(modality, source field)`):

- **Task 1** — `Any → 7T`: 4 pairs × 3 modalities = **12** `(modality, pair)` directories. Each contains `pred/` + `seg/` (both **mandatory**). Missing `seg/` causes Dice / Volume to be filled with worst-case (0.0).
- **Task 2** — `0.1T → Higher`: same shape as Task 1 — 4 pairs × 3 modalities = **12** directories, each with **mandatory** `pred/` + `seg/`.
- **Task 3** — `Any → Any` (single unified model): 20 directed pairs (incl. downfield, e.g. `7T_to_0.1T`) × 3 modalities = **60** directories. Each contains `pred/` **only** — Task 3 is scored on voxel-level metrics (nRMSE / SSIM / LPIPS) and **does not** use segmentation. Any `seg/` placed inside `task3.zip` is ignored.

Partial submissions are accepted: omit any modality / pair / subject you don't predict. Missing predictions are filled with worst-case metric values rather than skipped — see [evaluation-2026/README.md](evaluation-2026/README.md) "Missing-submission handling" for the exact arithmetic.

### Building the zips

This repo ships [submission_pack/](submission_pack/) as a starter template — three subtrees (`task1/`, `task2/`, `task3/`) already laid out with empty 0-byte `.nii.gz` placeholders so you see the exact filenames the scorer expects. Replace each placeholder with your model's prediction, then zip each task subtree:

```bash
# cd into the task subtree first, then zip its modality folders.
# This keeps the zip's internal root at the modality level (no `task{N}/` prefix).
cd Submission/submission_pack/task1 && zip -r ~/task1.zip T1W T2W T2FLAIR
cd Submission/submission_pack/task2 && zip -r ~/task2.zip T1W T2W T2FLAIR
cd Submission/submission_pack/task3 && zip -r ~/task3.zip T1W T2W T2FLAIR
```

> Note: the local directory name `submission_pack/` is just a working folder in this repo — it is **not** part of the submission and does **not** appear inside any uploaded zip.

### File Format & Naming

Predictions must be NIfTI (`.nii.gz`), float32, 364x436x364, intensity in [0, 1].

Filenames must match the corresponding ground-truth file:

| Type | Pattern | Example |
|------|---------|---------|
| Prediction | `P_{MOD}_{TARGET_FIELD}_{ID:04d}.nii.gz` | `P_T1W_7T_0001.nii.gz` |
| Segmentation (Task 1/2 only) | `P_{MOD}_{TARGET_FIELD}_{ID:04d}_seg.nii.gz` | `P_T1W_7T_0001_seg.nii.gz` |

- `MOD` ∈ {T1W, T2W, T2FLAIR}
- `TARGET_FIELD` is the **target** field of the pair (e.g. `7T` for `0.1T_to_7T`), not the input
- `ID` is the 4-digit zero-padded subject ID — copy it verbatim from the source input filename


## Testing Phase (To be updated)

*Docker container submission details will be provided here.*

## Contact

For submission issues: **mrixfields@outlook.com**
