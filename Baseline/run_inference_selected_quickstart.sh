#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MRIxFields2026 inference runner
#
# 推荐用法：
#   ./run_inference_selected.sh --tasks all --mode retro_scratch --gpu 0
#   ./run_inference_selected.sh --tasks 1,2 --mode retro_scratch --gpu 0
#   ./run_inference_selected.sh --tasks 3 --mode retro_scratch --gpu 0 --target-field 7T
#   ./run_inference_selected.sh --tasks all --mode pro_pretrained --gpu 0
#   ./run_inference_selected.sh --tasks all --dry-run
# 可改source/target fields:
# ./run_inference_selected_quickstart.sh --tasks 1 --mode retro_scratch --gpu 0 --source-field 3T --task1-target 7T --modality T1W
# ./run_inference_selected_quickstart.sh --tasks 2 --mode retro_scratch --gpu 0 --source-field 0.1T --task2-target 5T --modality T1W
# ./run_inference_selected_quickstart.sh --tasks 3 --mode retro_scratch --gpu 0 --source-field 5T --target-field 1.5T --modality T1W

#
# 注意：
#   这里的 --mode 是 checkpoint 所在目录名，不是 inference.py 的参数。
#   inference.py 本身不需要 --mode。
# ============================================================

TASKS="all"
MODE="retro_scratch"
GPU="0"
MODALITY="T1W"
SOURCE_FIELD="0.1T"
TASK1_TARGET_FIELD="7T"
TASK2_TARGET_FIELD="3T"
TASK3_TARGET_FIELD="7T"

EPOCH="100"
STARGAN_STEP="500000"

DRY_RUN=0
ALLOW_MISSING=0

# 可以手动覆盖 Task3 config / run name
TASK3_CONFIG_OVERRIDE=""
TASK3_RUN_NAME_OVERRIDE=""

print_help() {
    echo ""
    echo "Usage:"
    echo "  $0 --tasks all --mode retro_scratch --gpu 0"
    echo "  $0 --tasks 1,2 --mode retro_scratch --gpu 0"
    echo "  $0 --tasks 3 --mode retro_scratch --target-field 7T"
    echo "  $0 --tasks all --mode pro_pretrained --gpu 0"
    echo "  $0 --tasks all --dry-run"
    echo ""
    echo "Options:"
    echo "  --tasks              1 / 2 / 3 / 1,2 / 1,2,3 / all"
    echo "  --mode               retro_scratch / pro_pretrained / pro_scratch"
    echo "  --gpu                GPU id for inference, default: 0"
    echo "  --modality           T1W / T2W / T2FLAIR, default: T1W"
    echo "  --source-field       input field, default: 0.1T"
    echo "  --task1-target       default: 7T"
    echo "  --task2-target       default: 3T"
    echo "  --target-field       Task3 target field, default: 7T"
    echo "  --epoch              checkpoint_epochXXX.pth for CUT/CycleGAN, default: 100"
    echo "  --stargan-step       checkpoint_XXXXXX.pth for StarGAN, default: 500000"
    echo "  --task3-config       manually set Task3 config"
    echo "  --task3-run-name     manually set Task3 output run name under OUTPUT_DIR"
    echo "  --dry-run            print commands only"
    echo "  --allow-missing      skip task if checkpoint/config/input missing"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --modality)
            MODALITY="$2"
            shift 2
            ;;
        --source-field)
            SOURCE_FIELD="$2"
            shift 2
            ;;
        --task1-target)
            TASK1_TARGET_FIELD="$2"
            shift 2
            ;;
        --task2-target)
            TASK2_TARGET_FIELD="$2"
            shift 2
            ;;
        --target-field)
            TASK3_TARGET_FIELD="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --stargan-step)
            STARGAN_STEP="$2"
            shift 2
            ;;
        --task3-config)
            TASK3_CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --task3-run-name)
            TASK3_RUN_NAME_OVERRIDE="$2"
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

if [[ "$TASKS" == "all" ]]; then
    TASKS="1,2,3"
fi

if [[ "$MODE" != "retro_scratch" && "$MODE" != "pro_pretrained" && "$MODE" != "pro_scratch" ]]; then
    echo "[ERROR] --mode must be one of: retro_scratch / pro_pretrained / pro_scratch"
    exit 1
fi

# ------------------------------------------------------------
# Resolve repo root and .env paths safely
# ------------------------------------------------------------
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

mkdir -p "${INFERENCE_DIR_ABS}"
mkdir -p "${BASELINE_DIR}/logs"

cd "${BASELINE_DIR}"

find_latest_ckpt() {
    local weights_dir="$1"
    local expected="$2"

    if [[ -f "$expected" ]]; then
        echo "$expected"
        return 0
    fi

    if [[ -d "$weights_dir" ]]; then
        local latest
        latest="$(find "$weights_dir" -maxdepth 1 -type f \( -name 'checkpoint_epoch*.pth' -o -name 'checkpoint_*.pth' \) | sort -V | tail -n 1 || true)"
        if [[ -n "$latest" ]]; then
            echo "$latest"
            return 0
        fi
    fi

    return 1
}

check_or_skip() {
    local path="$1"
    local what="$2"
    local task="$3"

    if [[ -e "$path" ]]; then
        return 0
    fi

    if [[ "$ALLOW_MISSING" -eq 1 ]]; then
        echo "[WARN] Missing ${what} for Task ${task}: ${path}"
        echo "[WARN] Skipping Task ${task}"
        return 1
    fi

    echo "[ERROR] Missing ${what} for Task ${task}: ${path}"
    exit 1
}

run_cmd() {
    local log_file="$1"
    shift

    echo ""
    echo "------------------------------------------------------------"
    echo "Command:"
    printf ' %q' "$@"
    echo ""
    echo "Log: ${log_file}"
    echo "------------------------------------------------------------"
    echo ""

    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi

    "$@" 2>&1 | tee "$log_file"
}

run_task1() {
    local config="configs/task1/cut/${SOURCE_FIELD}_to_${TASK1_TARGET_FIELD}_${MODALITY}.yaml"
    local run_name="task1_${SOURCE_FIELD}_to_${TASK1_TARGET_FIELD}_${MODALITY}"
    local method="cut"

    local input_dir="${DATA_DIR_ABS}/Validating_prospective/${MODALITY}/${SOURCE_FIELD}"
    local output_dir="${INFERENCE_DIR_ABS}/task1_${SOURCE_FIELD}_to_${TASK1_TARGET_FIELD}_${MODALITY}/${MODE}"
    local weights_dir="${OUTPUT_DIR_ABS}/${run_name}/${method}/${MODE}/weights"
    local expected_ckpt="${weights_dir}/checkpoint_epoch${EPOCH}.pth"
    local checkpoint

    check_or_skip "$config" "config" "1" || return 0
    check_or_skip "$input_dir" "input_dir" "1" || return 0

    if ! checkpoint="$(find_latest_ckpt "$weights_dir" "$expected_ckpt")"; then
        check_or_skip "$expected_ckpt" "checkpoint" "1" || return 0
    fi

    mkdir -p "$output_dir"

    run_cmd "logs/infer_task1_${SOURCE_FIELD}_to_${TASK1_TARGET_FIELD}_${MODALITY}_${MODE}.log" \
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
        python scripts/inference.py \
            --config "$config" \
            --checkpoint "$checkpoint" \
            --input_dir "$input_dir" \
            --output_dir "$output_dir"
}

run_task2() {
    local config="configs/task2/cyclegan/${SOURCE_FIELD}_to_${TASK2_TARGET_FIELD}_${MODALITY}.yaml"
    local run_name="task2_${SOURCE_FIELD}_to_${TASK2_TARGET_FIELD}_${MODALITY}"
    local method="cyclegan"

    local input_dir="${DATA_DIR_ABS}/Validating_prospective/${MODALITY}/${SOURCE_FIELD}"
    local output_dir="${INFERENCE_DIR_ABS}/task2_${SOURCE_FIELD}_to_${TASK2_TARGET_FIELD}_${MODALITY}/${MODE}"
    local weights_dir="${OUTPUT_DIR_ABS}/${run_name}/${method}/${MODE}/weights"
    local expected_ckpt="${weights_dir}/checkpoint_epoch${EPOCH}.pth"
    local checkpoint

    check_or_skip "$config" "config" "2" || return 0
    check_or_skip "$input_dir" "input_dir" "2" || return 0

    if ! checkpoint="$(find_latest_ckpt "$weights_dir" "$expected_ckpt")"; then
        check_or_skip "$expected_ckpt" "checkpoint" "2" || return 0
    fi

    mkdir -p "$output_dir"

    run_cmd "logs/infer_task2_${SOURCE_FIELD}_to_${TASK2_TARGET_FIELD}_${MODALITY}_${MODE}.log" \
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
        python scripts/inference.py \
            --config "$config" \
            --checkpoint "$checkpoint" \
            --input_dir "$input_dir" \
            --output_dir "$output_dir"
}

resolve_task3_config() {
    if [[ -n "$TASK3_CONFIG_OVERRIDE" ]]; then
        echo "$TASK3_CONFIG_OVERRIDE"
        return 0
    fi

    if [[ -f "configs/task3/stargan/any_to_any_${MODALITY}.yaml" ]]; then
        echo "configs/task3/stargan/any_to_any_${MODALITY}.yaml"
        return 0
    fi

    echo "configs/task3/stargan/any_to_any_all_modalities.yaml"
}

resolve_task3_run_name() {
    if [[ -n "$TASK3_RUN_NAME_OVERRIDE" ]]; then
        echo "$TASK3_RUN_NAME_OVERRIDE"
        return 0
    fi

    if [[ -d "${OUTPUT_DIR_ABS}/task3_any_to_any_${MODALITY}/stargan_v2/${MODE}/weights" ]]; then
        echo "task3_any_to_any_${MODALITY}"
        return 0
    fi

    echo "task3_any_to_any_multimodal"
}

run_task3() {
    local config
    local run_name
    config="$(resolve_task3_config)"
    run_name="$(resolve_task3_run_name)"

    local method="stargan_v2"
    local input_dir="${DATA_DIR_ABS}/Validating_prospective/${MODALITY}/${SOURCE_FIELD}"
    local output_dir="${INFERENCE_DIR_ABS}/task3_${SOURCE_FIELD}_to_${TASK3_TARGET_FIELD}_${MODALITY}/${MODE}"

    local weights_dir="${OUTPUT_DIR_ABS}/${run_name}/${method}/${MODE}/weights"
    local expected_ckpt="${weights_dir}/checkpoint_${STARGAN_STEP}.pth"
    local checkpoint

    check_or_skip "$config" "config" "3" || return 0
    check_or_skip "$input_dir" "input_dir" "3" || return 0

    if ! checkpoint="$(find_latest_ckpt "$weights_dir" "$expected_ckpt")"; then
        check_or_skip "$expected_ckpt" "checkpoint" "3" || return 0
    fi

    mkdir -p "$output_dir"

    run_cmd "logs/infer_task3_${SOURCE_FIELD}_to_${TASK3_TARGET_FIELD}_${MODALITY}_${MODE}.log" \
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
        python scripts/inference.py \
            --config "$config" \
            --checkpoint "$checkpoint" \
            --input_dir "$input_dir" \
            --output_dir "$output_dir" \
            --target_field "$TASK3_TARGET_FIELD"
}

echo ""
echo "============================================================"
echo "Inference settings"
echo "Tasks:          ${TASKS}"
echo "Mode:           ${MODE}"
echo "GPU:            ${GPU}"
echo "Modality:       ${MODALITY}"
echo "Source field:   ${SOURCE_FIELD}"
echo "Task1 target:   ${TASK1_TARGET_FIELD}"
echo "Task2 target:   ${TASK2_TARGET_FIELD}"
echo "Task3 target:   ${TASK3_TARGET_FIELD}"
echo "DATA_DIR:       ${DATA_DIR_ABS}"
echo "OUTPUT_DIR:     ${OUTPUT_DIR_ABS}"
echo "INFERENCE_DIR:  ${INFERENCE_DIR_ABS}"
echo "Dry run:        ${DRY_RUN}"
echo "============================================================"
echo ""

IFS=',' read -ra TASK_ARRAY <<< "$TASKS"

for task in "${TASK_ARRAY[@]}"; do
    task="$(echo "$task" | xargs)"

    case "$task" in
        1) run_task1 ;;
        2) run_task2 ;;
        3) run_task3 ;;
        *)
            echo "[ERROR] Unknown task: $task"
            exit 1
            ;;
    esac
done

echo ""
echo "All selected inference jobs finished."
