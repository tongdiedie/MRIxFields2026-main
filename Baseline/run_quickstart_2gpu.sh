#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MRIxFields2026 baseline runner
# 每次用两张 GPU 跑一个 task，task 之间顺序执行
#
# 用法示例：
#   ./run_selected_tasks_2gpu.sh --tasks 1 --modes retro
#   ./run_selected_tasks_2gpu.sh --tasks 1,2 --modes retro
#   ./run_selected_tasks_2gpu.sh --tasks 1,2,3 --modes retro
#   ./run_selected_tasks_2gpu.sh --tasks all --modes retro
#   ./run_selected_tasks_2gpu.sh --tasks all --modes pro
#   ./run_selected_tasks_2gpu.sh --tasks all --modes full
#
# modes:
#   retro = 只跑 retro_scratch
#   full  = 跑 retro_scratch + pro_pretrained
# ============================================================


GPUS="0,1"
NPROC=2
TASKS="all"
MODES="retro"
MASTER_PORT=29501

CONFIG_TASK1="configs/task1/cut/0.1T_to_7T_T1W.yaml"
CONFIG_TASK2="configs/task2/cyclegan/0.1T_to_3T_T1W.yaml"
CONFIG_TASK3="configs/task3/stargan/any_to_any_T1W.yaml"

mkdir -p logs

print_help() {
    echo ""
    echo "Usage:"
    echo "  $0 --tasks 1 --modes retro"
    echo "  $0 --tasks 1,2 --modes retro"
    echo "  $0 --tasks all --modes retro"
    echo "  $0 --tasks all --modes full"
    echo "  $0 --tasks all --modes pro"
    echo ""
    echo "Options:"
    echo "  --tasks  1 / 2 / 3 / 1,2 / 1,3 / 2,3 / 1,2,3 / all"
    echo "  --modes  retro / pro / full"
    echo "  --gpus   default: 0,1"
    echo "  --port   default: 29501"
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
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
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

    echo ""
    echo "============================================================"
    echo "START"
    echo "Task:   ${task}"
    echo "Mode:   ${mode}"
    echo "Config: ${config}"
    echo "GPUs:   ${GPUS}"
    echo "Log:    logs/${tag}_${mode}_2gpu.log"
    echo "============================================================"
    echo ""

    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NPROC}" \
        --master_port="${MASTER_PORT}" \
        scripts/train.py \
        --config "${config}" \
        --mode "${mode}" \
        --dist \
        2>&1 | tee "logs/${tag}_${mode}_2gpu.log"

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
    *)
        echo "[ERROR] --modes only supports: retro / pro / full"
        exit 1
        ;;
esac

echo ""
echo "All selected jobs finished."
