#!/usr/bin/env bash
set -euo pipefail

GPU="0"
MODE="retro_scratch" # retro_scratch / pro_pretrained / pro_scratch
METHOD="cyclegan"
MODALITY="T1W"
SOURCE_FIELD="0.1T"
TARGETS="1.5T,3T,5T,7T"
EPOCH="100"
DRY_RUN=0
ALLOW_MISSING=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="$2"; shift 2 ;;
        --mode)
            MODE="$2"; shift 2 ;;
        --method)
            METHOD="$2"; shift 2 ;;
        --modality)
            MODALITY="$2"; shift 2 ;;
        --targets)
            TARGETS="$2"; shift 2 ;;
        --epoch)
            EPOCH="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --allow-missing)
            ALLOW_MISSING=1; shift ;;
        -h|--help)
            echo "Usage:"
            echo "  $0 --gpu 0 --mode retro_scratch --method cyclegan --targets 1.5T,3T,5T,7T"
            exit 0 ;;
        *)
            echo "[ERROR] Unknown arg: $1"
            exit 1 ;;
    esac
done

BASELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${BASELINE_DIR}/.." && pwd)"

set -a
source "${REPO_ROOT}/.env"
set +a

to_abs_from_root() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        p="${p#./}"
        echo "${REPO_ROOT}/${p}"
    fi
}

DATA_DIR_ABS="$(to_abs_from_root "${DATA_DIR}")"
OUTPUT_DIR_ABS="$(to_abs_from_root "${OUTPUT_DIR}")"
INFERENCE_DIR_ABS="$(to_abs_from_root "${INFERENCE_DIR}")"

cd "${BASELINE_DIR}"
mkdir -p logs
mkdir -p "${INFERENCE_DIR_ABS}"

IFS=',' read -ra TARGET_ARRAY <<< "${TARGETS}"

for tgt in "${TARGET_ARRAY[@]}"; do
    tgt="$(echo "$tgt" | xargs)"

    CONFIG="configs/task2/${METHOD}/${SOURCE_FIELD}_to_${tgt}_${MODALITY}.yaml"
    RUN_NAME="task2_${SOURCE_FIELD}_to_${tgt}_${MODALITY}"
    INPUT_DIR="${DATA_DIR_ABS}/Validating_prospective/${MODALITY}/${SOURCE_FIELD}"
    OUTPUT_DIR="${INFERENCE_DIR_ABS}/task2_${SOURCE_FIELD}_to_${tgt}_${MODALITY}/${METHOD}/${MODE}"
    WEIGHTS_DIR="${OUTPUT_DIR_ABS}/${RUN_NAME}/${METHOD}/${MODE}/weights"
    CKPT="${WEIGHTS_DIR}/checkpoint_epoch${EPOCH}.pth"

    if [[ ! -f "$CONFIG" ]]; then
        msg="[WARN] config not found: $CONFIG"
        [[ "$ALLOW_MISSING" -eq 1 ]] && echo "$msg, skip" && continue
        echo "[ERROR] $msg"; exit 1
    fi

    if [[ ! -d "$INPUT_DIR" ]]; then
        msg="[WARN] input_dir not found: $INPUT_DIR"
        [[ "$ALLOW_MISSING" -eq 1 ]] && echo "$msg, skip" && continue
        echo "[ERROR] $msg"; exit 1
    fi

    if [[ ! -f "$CKPT" ]]; then
        LATEST="$(find "$WEIGHTS_DIR" -maxdepth 1 -type f -name 'checkpoint_epoch*.pth' 2>/dev/null | sort -V | tail -n 1 || true)"
        if [[ -n "$LATEST" ]]; then
            CKPT="$LATEST"
        else
            msg="[WARN] checkpoint not found: ${WEIGHTS_DIR}"
            [[ "$ALLOW_MISSING" -eq 1 ]] && echo "$msg, skip" && continue
            echo "[ERROR] $msg"; exit 1
        fi
    fi

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "============================================================"
    echo "Task2 inference: ${SOURCE_FIELD} -> ${tgt}, ${MODALITY}, ${METHOD}, ${MODE}"
    echo "Config:     ${CONFIG}"
    echo "Checkpoint: ${CKPT}"
    echo "Input:      ${INPUT_DIR}"
    echo "Output:     ${OUTPUT_DIR}"
    echo "============================================================"

    CMD=(
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}"
        python scripts/inference.py
        --config "${CONFIG}"
        --checkpoint "${CKPT}"
        --input_dir "${INPUT_DIR}"
        --output_dir "${OUTPUT_DIR}"
    )

    printf ' %q' "${CMD[@]}"
    echo ""

    if [[ "$DRY_RUN" -eq 0 ]]; then
        "${CMD[@]}" 2>&1 | tee "logs/infer_task2_${SOURCE_FIELD}_to_${tgt}_${MODALITY}_${METHOD}_${MODE}.log"
    fi
done

echo ""
echo "Task2 multi-target inference finished."
