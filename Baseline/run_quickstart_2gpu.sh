#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MRIxFields2026 quickstart 2-GPU DDP runner
#
# 每次用 GPU0+GPU1 跑一个 task。
# Task 之间顺序执行。
#
# 用法：
#   ./run_quickstart_2gpu.sh --tasks 1 --modes retro
#   ./run_quickstart_2gpu.sh --tasks 1,2 --modes retro
#   ./run_quickstart_2gpu.sh --tasks all --modes retro
#   ./run_quickstart_2gpu.sh --tasks all --modes pro
#   ./run_quickstart_2gpu.sh --tasks all --modes full
#
# modes:
#   retro = retro_scratch
#   pro   = pro_pretrained
#   full  = 先 retro_scratch，再 pro_pretrained
# ============================================================

GPUS="0,1"
TASKS="all"
MODES="retro"
MASTER_PORT=29531
TRAIN_SCRIPT="scripts/train_with_eta.py"

# Quick Start configs
CONFIG_TASK1="configs/task1/cut/0.1T_to_7T_T1W.yaml"
CONFIG_TASK2="configs/task2/cyclegan/0.1T_to_3T_T1W.yaml"
CONFIG_TASK3="configs/task3/stargan/any_to_any_T1W.yaml"

# 默认不要开 INFO，否则 NCCL 日志会刷屏。
# 如果要 debug，可以运行前设置：export NCCL_DEBUG=INFO
NCCL_DEBUG_LEVEL="${NCCL_DEBUG:-WARN}"

mkdir -p logs

count_gpus() {
    local s="$1"
    awk -F',' '{print NF}' <<< "$s"
}

NPROC="$(count_gpus "$GPUS")"

print_help() {
    echo ""
    echo "Usage:"
    echo "  $0 --tasks 1 --modes retro"
    echo "  $0 --tasks 1,2 --modes retro"
    echo "  $0 --tasks all --modes retro"
    echo "  $0 --tasks all --modes pro"
    echo "  $0 --tasks all --modes full"
    echo ""
    echo "Options:"
    echo "  --tasks         1 / 2 / 3 / 1,2 / 1,2,3 / all"
    echo "  --modes         retro / pro / full"
    echo "  --gpus          default: 0,1"
    echo "  --port          default: 29531"
    echo "  --train-script  default: scripts/train_with_eta.py"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            NPROC="$(count_gpus "$GPUS")"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --train-script)
            TRAIN_SCRIPT="$2"
            shift 2
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

if [[ "$MODES" != "retro" && "$MODES" != "pro" && "$MODES" != "full" ]]; then
    echo "[ERROR] --modes only supports: retro / pro / full"
    exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "[ERROR] Train script not found: $TRAIN_SCRIPT"
    exit 1
fi

IFS=',' read -ra TASK_ARRAY <<< "$TASKS"

get_config() {
    local task="$1"
    case "$task" in
        1) echo "$CONFIG_TASK1" ;;
        2) echo "$CONFIG_TASK2" ;;
        3) echo "$CONFIG_TASK3" ;;
        *)
            echo "[ERROR] Unknown task: $task" >&2
            exit 1
            ;;
    esac
}

get_tag() {
    local task="$1"
    case "$task" in
        1) echo "task1_cut_0.1T_to_7T_T1W" ;;
        2) echo "task2_cyclegan_0.1T_to_3T_T1W" ;;
        3) echo "task3_stargan_any_to_any_T1W" ;;
        *)
            echo "[ERROR] Unknown task: $task" >&2
            exit 1
            ;;
    esac
}

run_one() {
    local task="$1"
    local mode="$2"

    task="$(echo "$task" | xargs)"

    local config
    local tag
    config="$(get_config "$task")"
    tag="$(get_tag "$task")"

    if [[ ! -f "$config" ]]; then
        echo "[ERROR] Config not found: $config"
        exit 1
    fi

    local log_file="logs/${tag}_${mode}_2gpu.log"

    echo ""
    echo "============================================================"
    echo "START"
    echo "Task:         ${task}"
    echo "Mode:         ${mode}"
    echo "Config:       ${config}"
    echo "Train script: ${TRAIN_SCRIPT}"
    echo "GPUs:         ${GPUS}"
    echo "NPROC:        ${NPROC}"
    echo "Port:         ${MASTER_PORT}"
    echo "Log:          ${log_file}"
    echo "============================================================"
    echo ""

    env \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="${GPUS}" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        OMP_NUM_THREADS=1 \
        NCCL_DEBUG="${NCCL_DEBUG_LEVEL}" \
        NCCL_IB_DISABLE=1 \
        NCCL_P2P_DISABLE=1 \
        NCCL_CUMEM_HOST_ENABLE=0 \
        TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node="${NPROC}" \
            --master_port="${MASTER_PORT}" \
            "${TRAIN_SCRIPT}" \
            --config "${config}" \
            --mode "${mode}" \
            --dist \
            2>&1 | tee "${log_file}"

    echo ""
    echo "============================================================"
    echo "DONE"
    echo "Task: ${task}"
    echo "Mode: ${mode}"
    echo "============================================================"
    echo ""
}

echo ""
echo "Selected tasks: ${TASKS}"
echo "Selected modes: ${MODES}"
echo "Using GPUs:     ${GPUS}"
echo "NPROC:          ${NPROC}"
echo "Train script:   ${TRAIN_SCRIPT}"
echo "NCCL_DEBUG:     ${NCCL_DEBUG_LEVEL}"
echo ""

case "$MODES" in
    retro)
        for task in "${TASK_ARRAY[@]}"; do
            run_one "$task" "retro_scratch"
        done
        ;;
    pro)
        for task in "${TASK_ARRAY[@]}"; do
            run_one "$task" "pro_pretrained"
        done
        ;;
    full)
        echo "Stage 1: running retro_scratch for selected tasks..."
        for task in "${TASK_ARRAY[@]}"; do
            run_one "$task" "retro_scratch"
        done

        echo "Stage 2: running pro_pretrained for selected tasks..."
        for task in "${TASK_ARRAY[@]}"; do
            run_one "$task" "pro_pretrained"
        done
        ;;
esac

echo ""
echo "All selected DDP jobs finished."
