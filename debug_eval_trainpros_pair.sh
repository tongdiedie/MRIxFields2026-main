#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Local debug evaluation using Training_prospective paired data
#
# Usage:
#   # 3 voxel metrics only
#   ./debug_eval_trainpros_pair.sh 2 cyclegan 0.1T 3T T1W retro_scratch 100 voxel
#
#   # all 5 metrics: nrmse ssim lpips dice volume
#   ./debug_eval_trainpros_pair.sh 2 cyclegan 0.1T 3T T1W retro_scratch 100 all5
#
#   # pro_pretrained example
#   ./debug_eval_trainpros_pair.sh 2 cyclegan 0.1T 3T T1W pro_pretrained 50 all5
#
# Args:
#   $1 task:       1 or 2
#   $2 method:     cut or cyclegan
#   $3 src field:  0.1T / 1.5T / 3T / 5T / 7T
#   $4 tgt field:  0.1T / 1.5T / 3T / 5T / 7T
#   $5 modality:   T1W / T2W / T2FLAIR
#   $6 mode:       retro_scratch / pro_pretrained / pro_scratch
#   $7 epoch:      checkpoint epoch, e.g. 100 or 50
#   $8 eval_mode:  voxel / all5
# ============================================================

TASK="${1:-2}"
METHOD="${2:-cyclegan}"
SRC_FIELD="${3:-0.1T}"
TGT_FIELD="${4:-3T}"
MODALITY="${5:-T1W}"
MODE="${6:-retro_scratch}"
EPOCH="${7:-100}"
EVAL_MODE="${8:-voxel}"

if [[ "$EVAL_MODE" != "voxel" && "$EVAL_MODE" != "all5" ]]; then
  echo "[ERROR] eval_mode must be voxel or all5"
  exit 1
fi

set -a
source .env
set +a

if [[ "$TASK" == "1" ]]; then
  TASK_NAME="task1_${SRC_FIELD}_to_${TGT_FIELD}_${MODALITY}"
  CONFIG="Baseline/configs/task1/${METHOD}/${SRC_FIELD}_to_${TGT_FIELD}_${MODALITY}.yaml"
elif [[ "$TASK" == "2" ]]; then
  TASK_NAME="task2_${SRC_FIELD}_to_${TGT_FIELD}_${MODALITY}"
  CONFIG="Baseline/configs/task2/${METHOD}/${SRC_FIELD}_to_${TGT_FIELD}_${MODALITY}.yaml"
else
  echo "[ERROR] This script supports Task1/Task2 only."
  exit 1
fi

CKPT="$OUTPUT_DIR/${TASK_NAME}/${METHOD}/${MODE}/weights/checkpoint_epoch${EPOCH}.pth"

TMP_INPUT="$INFERENCE_DIR/debug_input_trainpros_${TASK_NAME}_${METHOD}_${MODE}"
PRED_DIR="$INFERENCE_DIR/debug_pred_trainpros_${TASK_NAME}_${METHOD}_${MODE}"
PRED_SEG_DIR="${PRED_DIR}_seg"
TARGET_SEG_DIR="$INFERENCE_DIR/debug_target_seg_trainpros_${MODALITY}_${TGT_FIELD}"
MATCHED_IDS="$PRED_DIR/matched_ids.txt"

rm -rf "$TMP_INPUT" "$PRED_DIR"
mkdir -p "$TMP_INPUT" "$PRED_DIR"

echo "============================================================"
echo "Debug evaluation with Training_prospective"
echo "Task:       $TASK"
echo "Method:     $METHOD"
echo "Pair:       $SRC_FIELD -> $TGT_FIELD"
echo "Modality:   $MODALITY"
echo "Mode:       $MODE"
echo "Epoch:      $EPOCH"
echo "Eval mode:  $EVAL_MODE"
echo "Config:     $CONFIG"
echo "Checkpoint: $CKPT"
echo "Pred dir:   $PRED_DIR"
echo "============================================================"

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Config not found: $CONFIG"
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] Checkpoint not found: $CKPT"
  exit 1
fi

comm -12 \
  <(find "$DATA_DIR/Training_prospective/$MODALITY/$SRC_FIELD" -name "*.nii.gz" 2>/dev/null | sed -E 's/.*_([0-9]+)\.nii\.gz/\1/' | sort) \
  <(find "$DATA_DIR/Training_prospective/$MODALITY/$TGT_FIELD" -name "*.nii.gz" 2>/dev/null | sed -E 's/.*_([0-9]+)\.nii\.gz/\1/' | sort) \
  > "$MATCHED_IDS"

N=$(wc -l < "$MATCHED_IDS")
echo "Matched subjects: $N"
cat "$MATCHED_IDS"

if [[ "$N" -eq 0 ]]; then
  echo "[ERROR] No matched subjects for $SRC_FIELD -> $TGT_FIELD in Training_prospective/$MODALITY"
  exit 1
fi

while read id; do
  src_file="$DATA_DIR/Training_prospective/$MODALITY/$SRC_FIELD/P_${MODALITY}_${SRC_FIELD}_${id}.nii.gz"
  if [[ ! -f "$src_file" ]]; then
    echo "[WARN] Missing source file: $src_file"
    continue
  fi
  ln -s "$src_file" "$TMP_INPUT/P_${MODALITY}_${SRC_FIELD}_${id}.nii.gz"
done < "$MATCHED_IDS"

echo ""
echo "Running inference..."
python Baseline/scripts/inference.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --input_dir "$TMP_INPUT" \
  --output_dir "$PRED_DIR"

echo ""
echo "Checking prediction-target matched IDs..."
COMMON_AFTER_INFER=$(comm -12 \
  <(find "$PRED_DIR" -name "*.nii.gz" | sed -E 's/.*_([0-9]+)\.nii\.gz/\1/' | sort) \
  <(find "$DATA_DIR/Training_prospective/$MODALITY/$TGT_FIELD" -name "*.nii.gz" | sed -E 's/.*_([0-9]+)\.nii\.gz/\1/' | sort))

echo "$COMMON_AFTER_INFER"

if [[ -z "$COMMON_AFTER_INFER" ]]; then
  echo "[ERROR] No matched prediction-target IDs after inference."
  exit 1
fi

echo ""
echo "Running voxel-level evaluation..."
python Evaluation/evaluate.py \
  --pred_dir "$PRED_DIR" \
  --target_dir "$DATA_DIR/Training_prospective/$MODALITY/$TGT_FIELD/" \
  --metrics nrmse ssim lpips \
  --output_csv "$PRED_DIR/results_voxel.csv" \
  --output_json "$PRED_DIR/results_voxel.json"

if [[ "$EVAL_MODE" == "all5" ]]; then
  echo ""
  echo "Running SynthSeg for predictions..."
  rm -rf "$PRED_SEG_DIR"
  python Evaluation/segment.py \
    --input_dir "$PRED_DIR" \
    --output_dir "$PRED_SEG_DIR"

  echo ""
  echo "Running SynthSeg for target ground truth..."
  # 这里不强制删除 target seg，因为同一个 target field 可以复用。
  mkdir -p "$TARGET_SEG_DIR"
  python Evaluation/segment.py \
    --input_dir "$DATA_DIR/Training_prospective/$MODALITY/$TGT_FIELD/" \
    --output_dir "$TARGET_SEG_DIR"

  echo ""
  echo "Checking segmentation files..."
  echo "Prediction seg files:"
  find "$PRED_SEG_DIR" -name "*_seg.nii.gz" | sort

  echo "Target seg files:"
  find "$TARGET_SEG_DIR" -name "*_seg.nii.gz" | sort

  echo ""
  echo "Running all-5-metrics evaluation..."
  python Evaluation/evaluate.py \
    --pred_dir "$PRED_DIR" \
    --target_dir "$DATA_DIR/Training_prospective/$MODALITY/$TGT_FIELD/" \
    --pred_seg_dir "$PRED_SEG_DIR" \
    --target_seg_dir "$TARGET_SEG_DIR" \
    --metrics nrmse ssim lpips dice volume \
    --output_csv "$PRED_DIR/results_all5.csv" \
    --output_json "$PRED_DIR/results_all5.json"
fi

echo ""
echo "Saved:"
echo "  Predictions:       $PRED_DIR"
echo "  Voxel CSV:         $PRED_DIR/results_voxel.csv"
echo "  Voxel JSON:        $PRED_DIR/results_voxel.json"

if [[ "$EVAL_MODE" == "all5" ]]; then
  echo "  Prediction seg:    $PRED_SEG_DIR"
  echo "  Target seg:        $TARGET_SEG_DIR"
  echo "  All-5 CSV:         $PRED_DIR/results_all5.csv"
  echo "  All-5 JSON:        $PRED_DIR/results_all5.json"
fi
