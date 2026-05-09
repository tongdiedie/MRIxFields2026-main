#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Task3 StarGAN v2 Any -> Any inference
#
# 用法：
#   ./run_task3_any2any_inference.sh
#   ./run_task3_any2any_inference.sh --gpu 0
#   ./run_task3_any2any_inference.sh --modality T1W
#   ./run_task3_any2any_inference.sh --mode retro_scratch
#   ./run_task3_any2any_inference.sh --sources 0.1T,1.5T --targets 3T,7T
#   ./run_task3_any2any_inference.sh --sources 0.1T,1.5T,3T,5T,7T --targets 0.1T,1.5T,3T,5T,7T
#   ./run_task3_any2any_inference.sh --dry-run
# ============================================================

GPU="0"
MODE="retro_scratch"
MODALITY="T1W"
SOURCES="0.1T,1.5T,3T,5T,7T"
TARGETS="0.1T,1.5T,3T,5T,7T"
DRY_RUN=0
ALLOW_MISSING=0

CONFIG="configs/task3/stargan/any_to_any_T1W.yaml"
RUN_NAME="task3_any_to_any_T1W"
METHOD="stargan_v2"
STARGAN_STEP="500000"

print_help() {
    echo ""
    echo "Usage:"
    echo "  $0 --gpu 0 --mode retro_scratch --modality T1W"
    echo "  $0 --sources 0.1T,1.5T --targets 3T,7T"
    echo "  $0 --dry-run"
    echo ""
    echo "Options:"
    echo "  --gpu           GPU id, default: 0"
    echo "  --mode          retro_scratch / pro_pretrained / pro_scratch"
    echo "  --modality      T1W / T2W / T2FLAIR, default: T1W"
    echo "  --sources       comma-separated source fields"
    echo "  --targets       comma-separated target fields"
    echo "  --config        Task3 config path"
    echo "  --run-name      checkpoint run name under OUTPUT_DIR"
    echo "  --step          StarGAN checkpoint step, default: 500000"
    echo "  --dry-run       print commands only"
    echo "  --allow-missing skip missing input dirs/checkpoints"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --modality)
            MODALITY="$2"
            CONFIG="configs/task3/stargan/any_to_any_${MODALITY}.yaml"
            RUN_NAME="task3_any_to_any_${MODALITY}"
            shift 2
            ;;
        --sources)
            SOURCES="$2"
            shift 2
            ;;
        --targets)
            TARGETS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --step)
            STARGAN_STEP="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --allow-missing)
            ALLOW_MISSING=1
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            print_help
            exit 1
            ;;
    esac
done

BASELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${BASELINE_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
else
    echo "[ERROR] Cannot find ${REPO_ROOT}/.env"
    exit 1
fi

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

WEIGHTS_DIR="${OUTPUT_DIR_ABS}/${RUN_NAME}/${METHOD}/${MODE}/weights"
EXPECTED_CKPT="${WEIGHTS_DIR}/checkpoint_${STARGAN_STEP}.pth"

if [[ -f "${EXPECTED_CKPT}" ]]; then
    CHECKPOINT="${EXPECTED_CKPT}"
else
    CHECKPOINT="$(find "${WEIGHTS_DIR}" -maxdepth 1 -type f -name 'checkpoint_*.pth' 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config not found: ${CONFIG}"
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Checkpoint not found under: ${WEIGHTS_DIR}"
    echo "Expected: ${EXPECTED_CKPT}"
    exit 1
fi

echo ""
echo "============================================================"
echo "Task3 Any -> Any inference"
echo "Config:        ${CONFIG}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Mode:          ${MODE}"
echo "GPU:           ${GPU}"
echo "Modality:      ${MODALITY}"
echo "Sources:       ${SOURCES}"
echo "Targets:       ${TARGETS}"
echo "DATA_DIR:      ${DATA_DIR_ABS}"
echo "INFERENCE_DIR: ${INFERENCE_DIR_ABS}"
echo "Dry run:       ${DRY_RUN}"
echo "============================================================"
echo ""

IFS=',' read -ra SOURCE_ARRAY <<< "${SOURCES}"
IFS=',' read -ra TARGET_ARRAY <<< "${TARGETS}"

for src in "${SOURCE_ARRAY[@]}"; do
    src="$(echo "$src" | xargs)"

    INPUT_DIR="${DATA_DIR_ABS}/Validating_prospective/${MODALITY}/${src}"

    if [[ ! -d "${INPUT_DIR}" ]]; then
        if [[ "${ALLOW_MISSING}" -eq 1 ]]; then
            echo "[WARN] Missing input_dir, skip source ${src}: ${INPUT_DIR}"
            continue
        else
            echo "[ERROR] Missing input_dir: ${INPUT_DIR}"
            exit 1
        fi
    fi

    for tgt in "${TARGET_ARRAY[@]}"; do
        tgt="$(echo "$tgt" | xargs)"

        if [[ "${src}" == "${tgt}" ]]; then
            echo "[SKIP] source == target: ${src} -> ${tgt}"
            continue
        fi

        OUT_DIR="${INFERENCE_DIR_ABS}/task3_${src}_to_${tgt}_${MODALITY}/${MODE}"
        LOG_FILE="logs/infer_task3_${src}_to_${tgt}_${MODALITY}_${MODE}.log"

        mkdir -p "${OUT_DIR}"

        echo ""
        echo "------------------------------------------------------------"
        echo "Task3 inference: ${src} -> ${tgt}, ${MODALITY}"
        echo "Output: ${OUT_DIR}"
        echo "Log:    ${LOG_FILE}"
        echo "------------------------------------------------------------"

        CMD=(
            env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}"
            python scripts/inference.py
            --config "${CONFIG}"
            --checkpoint "${CHECKPOINT}"
            --input_dir "${INPUT_DIR}"
            --output_dir "${OUT_DIR}"
            --target_field "${tgt}"
        )

        printf ' %q' "${CMD[@]}"
        echo ""

        if [[ "${DRY_RUN}" -eq 0 ]]; then
            "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
        fi
    done
done

echo ""
echo "Task3 Any -> Any inference finished."
