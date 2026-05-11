# MRIxFields2026 MVP：FDS synthetic paired data + 3D/2.5D conditional U-Net

这个目录是一个**独立 MVP 代码包**，建议放在官方仓库根目录：

```text
MRIxFields2026/
├── Baseline/
├── Evaluation/
├── Submission/
├── .env
└── MVP/                 # 把本目录放在这里
```

它不修改官方 `Baseline/`。核心流程是：

1. 用 prospective paired 数据训练一个物理启发的 FDS：`high/target field -> low/source field`。
2. 用 FDS 把 retrospective target field 图像降级，生成 anatomy-paired synthetic source/target。
3. 用 real prospective pairs + synthetic pairs 训练一个 3D/2.5D field-conditioned U-Net。
4. 用官方 `Evaluation/evaluate.py` 做本地评估，用官方 `Evaluation/segment.py` 给 Task 1/2 生成 `seg/`。

---

## 0. 环境和路径

先按官方仓库要求创建环境，并在仓库根目录设置 `.env`：

```bash
cd /path/to/MRIxFields2026
conda env create -f environment.yml
conda activate mf
cp .env.example .env
# 编辑 DATA_DIR / OUTPUT_DIR / INFERENCE_DIR / SYNTHSEG_DIR / DEVICE
vim .env
```

下面所有命令默认从**官方仓库根目录**运行：

```bash
cd /path/to/MRIxFields2026
```

> 本代码会自动向上查找仓库根目录的 `.env`。配置文件里的 `./MVP/work/...`、`./MVP/runs/...` 都是相对仓库根目录解析的。

快速检查模型代码是否能 forward：

```bash
python MVP/scripts/quick_check.py
```

---

## 1. 生成 real paired manifest

Task 1：

```bash
mkdir -p MVP/work/manifests
python MVP/scripts/build_manifest.py \
  --task 1 \
  --split Training_prospective \
  --out_csv MVP/work/manifests/task1_real_train.csv
```

Task 2：

```bash
python MVP/scripts/build_manifest.py \
  --task 2 \
  --split Training_prospective \
  --out_csv MVP/work/manifests/task2_real_train.csv
```

Task 3：

```bash
python MVP/scripts/build_manifest.py \
  --task 3 \
  --split Training_prospective \
  --out_csv MVP/work/manifests/task3_real_train.csv
```

调试时可以加 `--modalities T1W --max_subjects 2`。

---

## 2. 训练 FDS degradation simulator

完整 downfield FDS：

```bash
python MVP/scripts/train_fds.py \
  --config MVP/configs/fds_all_downfield.yaml \
  --output_dir MVP/runs/fds_all_downfield
```

如果你只想先跑 Task 1，可以先只训练 7T 到其他 field 的 FDS：

```bash
python MVP/scripts/train_fds.py \
  --config MVP/configs/fds_all_downfield.yaml \
  --output_dir MVP/runs/fds_task1_7t_down \
  --pairs 7T:0.1T 7T:1.5T 7T:3T 7T:5T
```

FDS checkpoint 会保存在：

```text
MVP/runs/fds_all_downfield/best.pt
MVP/runs/fds_all_downfield/last.pt
```

---

## 3. 用 FDS 生成 synthetic paired data

Task 1 synthetic pairs：

```bash
python MVP/scripts/generate_synthetic_pairs.py \
  --checkpoint MVP/runs/fds_all_downfield/best.pt \
  --task 1 \
  --out_root MVP/work/synthetic_task1 \
  --amp
```

Task 2 synthetic pairs：

```bash
python MVP/scripts/generate_synthetic_pairs.py \
  --checkpoint MVP/runs/fds_all_downfield/best.pt \
  --task 2 \
  --out_root MVP/work/synthetic_task2 \
  --amp
```

Task 3 synthetic pairs：

```bash
python MVP/scripts/generate_synthetic_pairs.py \
  --checkpoint MVP/runs/fds_all_downfield/best.pt \
  --task 3 \
  --out_root MVP/work/synthetic_task3 \
  --amp
```

首次调试建议加 `--modalities T1W --limit_per_field 5`，确认流程无误后再全量生成。生成结束后会得到：

```text
MVP/work/synthetic_task1/synthetic_manifest_task1.csv
MVP/work/synthetic_task2/synthetic_manifest_task2.csv
MVP/work/synthetic_task3/synthetic_manifest_task3.csv
```

---

## 4. 训练 field-conditioned 3D/2.5D U-Net

Task 1：

```bash
python MVP/scripts/train_translator.py \
  --config MVP/configs/task1_mvp.yaml \
  --manifest \
    MVP/work/manifests/task1_real_train.csv \
    MVP/work/synthetic_task1/synthetic_manifest_task1.csv \
  --output_dir MVP/runs/task1_mvp
```

Task 2：

```bash
python MVP/scripts/train_translator.py \
  --config MVP/configs/task2_mvp.yaml \
  --manifest \
    MVP/work/manifests/task2_real_train.csv \
    MVP/work/synthetic_task2/synthetic_manifest_task2.csv \
  --output_dir MVP/runs/task2_mvp
```

Task 3：

```bash
python MVP/scripts/train_translator.py \
  --config MVP/configs/task3_mvp.yaml \
  --manifest \
    MVP/work/manifests/task3_real_train.csv \
    MVP/work/synthetic_task3/synthetic_manifest_task3.csv \
  --output_dir MVP/runs/task3_mvp
```

### 3D vs 2.5D

配置里的 patch size 是 `[D, H, W]`：

```yaml
data:
  patch_size: [64, 96, 96]
```

推荐起步：

- 显存较小：`[16, 192, 192]`，相当于 2.5D；
- 平衡版本：`[64, 96, 96]`；
- 冲榜训练：`[96, 96, 96]` 或更大，配合 gradient checkpoint/更强 GPU 后再加。

---

## 5. Inference：生成 submission-style prediction tree

Task 1 validation 示例，跑所有 modality 和 source field：

```bash
mkdir -p MVP/predictions/task1
for mod in T1W T2W T2FLAIR; do
  for src in 0.1T 1.5T 3T 5T; do
    python MVP/scripts/infer.py \
      --checkpoint MVP/runs/task1_mvp/best.pt \
      --input_dir "$DATA_DIR/Validating_prospective/${mod}/${src}" \
      --output_dir MVP/predictions/task1 \
      --source_field "$src" \
      --target_field 7T \
      --modality "$mod" \
      --amp
  done
done
```

Task 2 validation：

```bash
mkdir -p MVP/predictions/task2
for mod in T1W T2W T2FLAIR; do
  for tgt in 1.5T 3T 5T 7T; do
    python MVP/scripts/infer.py \
      --checkpoint MVP/runs/task2_mvp/best.pt \
      --input_dir "$DATA_DIR/Validating_prospective/${mod}/0.1T" \
      --output_dir MVP/predictions/task2 \
      --source_field 0.1T \
      --target_field "$tgt" \
      --modality "$mod" \
      --amp
  done
done
```

Task 3 validation：

```bash
mkdir -p MVP/predictions/task3
for mod in T1W T2W T2FLAIR; do
  for src in 0.1T 1.5T 3T 5T 7T; do
    for tgt in 0.1T 1.5T 3T 5T 7T; do
      [ "$src" = "$tgt" ] && continue
      python MVP/scripts/infer.py \
        --checkpoint MVP/runs/task3_mvp/best.pt \
        --input_dir "$DATA_DIR/Validating_prospective/${mod}/${src}" \
        --output_dir MVP/predictions/task3 \
        --source_field "$src" \
        --target_field "$tgt" \
        --modality "$mod" \
        --amp
    done
  done
done
```

输出结构会自动变成：

```text
MVP/predictions/task1/T1W/0.1T_to_7T/pred/P_T1W_7T_0001.nii.gz
```

---

## 6. Task 1/2：生成 SynthSeg `seg/`

Task 1/2 提交必须有 `seg/`。本脚本会对每个 `pred/` 目录调用官方 `Evaluation/segment.py`，并把结果放到同级 `seg/`：

```bash
python MVP/scripts/segment_submission_tree.py \
  --submission_root MVP/predictions/task1 \
  --task 1
```

Task 2：

```bash
python MVP/scripts/segment_submission_tree.py \
  --submission_root MVP/predictions/task2 \
  --task 2
```

Task 3 不需要 segmentation。

---

## 7. 用官方 evaluator 做本地评估

以 Task 1、T1W、0.1T→7T 为例：

```bash
# 先给 target GT 跑 SynthSeg，只需跑一次
python Evaluation/segment.py \
  --input_dir "$DATA_DIR/Validating_prospective/T1W/7T" \
  --output_dir MVP/work/gt_seg/T1W_7T

python MVP/scripts/run_official_eval.py \
  --task 1 \
  --pred_dir MVP/predictions/task1/T1W/0.1T_to_7T/pred \
  --target_dir "$DATA_DIR/Validating_prospective/T1W/7T" \
  --pred_seg_dir MVP/predictions/task1/T1W/0.1T_to_7T/seg \
  --target_seg_dir MVP/work/gt_seg/T1W_7T \
  --output_csv MVP/work/eval_task1_T1W_0.1T_to_7T.csv
```

Task 3 不需要 seg：

```bash
python MVP/scripts/run_official_eval.py \
  --task 3 \
  --pred_dir MVP/predictions/task3/T1W/0.1T_to_7T/pred \
  --target_dir "$DATA_DIR/Validating_prospective/T1W/7T" \
  --output_csv MVP/work/eval_task3_T1W_0.1T_to_7T.csv
```

---

## 8. 打包 submission zip

Task 1：

```bash
python MVP/scripts/package_submission.py \
  --task 1 \
  --pred_root MVP/predictions/task1 \
  --zip_path MVP/task1.zip
```

Task 2：

```bash
python MVP/scripts/package_submission.py \
  --task 2 \
  --pred_root MVP/predictions/task2 \
  --zip_path MVP/task2.zip
```

Task 3：

```bash
python MVP/scripts/package_submission.py \
  --task 3 \
  --pred_root MVP/predictions/task3 \
  --zip_path MVP/task3.zip
```

zip 内部 root 会直接是 `T1W/ T2W/ T2FLAIR/`，不会带 `MVP/` 或 `task1/` 前缀。

---

## 9. 重要调参建议

这个 MVP 的默认配置是能跑的基线，不是最终冲榜上限。建议按这个顺序优化：

1. **先跑单 modality、小 subject smoke run**：`--modalities T1W --max_subjects 2 --limit_per_field 5`。
2. **确认 FDS 生成的 synthetic source 视觉上像低场图**，但 anatomy 与 target 对齐。
3. **逐步加大 synthetic data 数量**：先 `limit_per_field=20`，再全量。
4. **Task 1/2 最后做 pair-specific fine-tune**：从 `task1_mvp/best.pt` 出发，只用某个 pair 的 manifest 继续训练 10–30 epoch。
5. **nRMSE/SSIM 优先时降低 LPIPS loss**；LPIPS 排名吃亏时再把 `loss.lpips` 从 `0.02` 调到 `0.03–0.05`。
6. **输出必须 clip 到 [0,1]**；本代码保存 NIfTI 时会自动 clip 并写 float32。

---

## 10. 文件说明

```text
MVP/
├── configs/
│   ├── fds_all_downfield.yaml
│   ├── task1_mvp.yaml
│   ├── task2_mvp.yaml
│   └── task3_mvp.yaml
├── mvp/
│   ├── conditioning.py       # log-field Fourier embedding + modality one-hot
│   ├── data.py               # manifest + 3D random patch dataset
│   ├── infer_utils.py        # 3D sliding-window inference
│   ├── io.py                 # NIfTI I/O + MRIxFields filename parsing
│   ├── losses.py             # L1 / nRMSE / differentiable SSIM / LPIPS-slice
│   ├── networks.py           # FDS + field-conditioned U-Net
│   ├── paths.py              # .env loader
│   └── train_utils.py
└── scripts/
    ├── build_manifest.py
    ├── train_fds.py
    ├── generate_synthetic_pairs.py
    ├── train_translator.py
    ├── infer.py
    ├── segment_submission_tree.py
    ├── run_official_eval.py
    ├── package_submission.py
    └── quick_check.py
```
