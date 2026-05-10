#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MRIxFields2026 full baseline training runner
#
# 每次用两张 GPU 跑一个 config，所有 config 顺序执行。
#
# 用法示例：
#   ./run_all_configs_2gpu.sh --tasks all --modes retro
#   ./run_all_configs_2gpu.sh --tasks 1 --modes retro
#   ./run_all_configs_2gpu.sh --tasks 1,2 --modes retro
#   ./run_all_configs_2gpu.sh --tasks all --modes full
#   ./run_all_configs_2gpu.sh --tasks all --modes pro
#   ./run_all_configs_2gpu.sh --tasks 2 --methods cyclegan --modalities T1W --modes retro
#   ./run_all_configs_2gpu.sh --tasks all --modes retro --dry-run
#   ./run_all_configs_2gpu.sh --tasks all --modes full
#   ./run_all_configs_2gpu.sh --dry-run
#
# modes:
#   retro = 只跑 retro_scratch
#   pro   = 只跑 pro_pretrained
#   full  = 先跑所有 retro_scratch，再跑所有 pro_pretrained
#
# 注意：
#   pro_pretrained 依赖对应 config 的 retro_scratch checkpoint。
# ============================================================

GPUS="0,1"
TASKS="all"
MODES="retro"
METHODS="cut,cyclegan"
MODALITIES="T1W,T2W,T2FLAIR"
MASTER_PORT=29501
DRY_RUN=0
ALLOW_MISSING=0

# 可选：如果 checkpoint 已经存在，就跳过。
# 默认不跳过，因为中途训练可能有旧 checkpoint 但还没完整完成。
SKIP_DONE=0

# 用于 skip-done 判断的默认 checkpoint。
# 如果你的 config 不是这些 epoch/step，可以不用 --skip-done。
PRETRAIN_EPOCH=100
FINETUNE_EPOCH=50
STARGAN_PRETRAIN_STEP=500000

mkdir -p logs

print_help() {
    echo ""
    echo "Usage:"
    echo "  $0 --tasks all --modes retro"
    echo "  $0 --tasks all --modes full"
    echo "  $0 --tasks 1,2 --modes retro"
    echo "  $0 --tasks 2 --methods cyclegan --modalities T1W --modes retro"
    echo "  $0 --dry-run"
    echo ""
    echo "Options:"
    echo "  --tasks            1 / 2 / 3 / 1,2 / 1,2,3 / all"
    echo "  --modes            retro / pro / full"
    echo "  --methods          cut / cyclegan / cut,cyclegan / all"
    echo "  --modalities       T1W / T2W / T2FLAIR / comma-separated / all"
    echo "  --gpus             default: 0,1"
    echo "  --port             default: 29501"
    echo "  --dry-run          print commands only"
    echo "  --allow-missing    skip missing config files"
    echo "  --skip-done        skip if expected final checkpoint exists"
    echo "  --pretrain-epoch   default: 100"
    echo "  --finetune-epoch   default: 50"
    echo "  --stargan-step     default: 500000"
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
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --modalities)
            MODALITIES="$2"
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
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --allow-missing)
            ALLOW_MISSING=1
            shift
            ;;
        --skip-done)
            SKIP_DONE=1
            shift
            ;;
        --pretrain-epoch)
            PRETRAIN_EPOCH="$2"
            shift 2
            ;;
        --finetune-epoch)
            FINETUNE_EPOCH="$2"
            shift 2
            ;;
        --stargan-step)
            STARGAN_PRETRAIN_STEP="$2"
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

if [[ "$METHODS" == "all" ]]; then
    METHODS="cut,cyclegan"
fi

if [[ "$MODALITIES" == "all" ]]; then
    MODALITIES="T1W,T2W,T2FLAIR"
fi

if [[ "$MODES" != "retro" && "$MODES" != "pro" && "$MODES" != "full" ]]; then
    echo "[ERROR] --modes only supports: retro / pro / full"
    exit 1
fi

IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
IFS=',' read -ra MODALITY_ARRAY <<< "$MODALITIES"

count_gpus() {
    local s="$1"
    awk -F',' '{print NF}' <<< "$s"
}

NPROC="$(count_gpus "$GPUS")"

trim() {
    echo "$1" | xargs
}

config_exists_or_handle() {
    local cfg="$1"

    if [[ -f "$cfg" ]]; then
        return 0
    fi

    if [[ "$ALLOW_MISSING" -eq 1 ]]; then
        echo "[WARN] Missing config, skip: $cfg"
        return 1
    fi

    echo "[ERROR] Config not found: $cfg"
    echo "Use --allow-missing if you want to skip missing configs."
    exit 1
}

sanitize_tag() {
    local cfg="$1"
    local mode="$2"
    local tag="${cfg#configs/}"
    tag="${tag%.yaml}"
    tag="${tag//\//_}"
    tag="${tag//./p}"
    echo "${tag}_${mode}_2gpu"
}

get_task_name_from_config() {
    local cfg="$1"
    python - "$cfg" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
print(cfg.get("task_name", "unknown_task"))
PY
}

get_method_from_config() {
    local cfg="$1"
    python - "$cfg" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
print(cfg.get("method", "unknown_method"))
PY
}

expected_ckpt_for_config_mode() {
    local cfg="$1"
    local mode="$2"

    local task_name
    local method
    task_name="$(get_task_name_from_config "$cfg")"
    method="$(get_method_from_config "$cfg")"

    # 读取 .env 里的 OUTPUT_DIR；如果没有，就用 Baseline/runs。
    local output_dir="${OUTPUT_DIR:-./Baseline/runs}"
    if [[ "$output_dir" != /* ]]; then
        output_dir="../${output_dir#./}"
    fi

    local weights_dir="${output_dir}/${task_name}/${method}/${mode}/weights"

    if [[ "$method" == "stargan_v2" && "$mode" == "retro_scratch" ]]; then
        echo "${weights_dir}/checkpoint_${STARGAN_PRETRAIN_STEP}.pth"
    else
        if [[ "$mode" == "retro_scratch" ]]; then
            echo "${weights_dir}/checkpoint_epoch${PRETRAIN_EPOCH}.pth"
        else
            echo "${weights_dir}/checkpoint_epoch${FINETUNE_EPOCH}.pth"
        fi
    fi
}

should_skip_done() {
    local cfg="$1"
    local mode="$2"

    if [[ "$SKIP_DONE" -eq 0 ]]; then
        return 1
    fi

    local ckpt
    ckpt="$(expected_ckpt_for_config_mode "$cfg" "$mode")"

    if [[ -f "$ckpt" ]]; then
        echo "[SKIP-DONE] Found checkpoint: $ckpt"
        return 0
    fi

    return 1
}

build_config_list() {
    CONFIGS=()

    for task_raw in "${TASK_ARRAY[@]}"; do
        task="$(trim "$task_raw")"

        case "$task" in
            1)
                # Task1: Any -> 7T
                # sources: 0.1T / 1.5T / 3T / 5T
                local sources=("0.1T" "1.5T" "3T" "5T")
                local target="7T"

                for method_raw in "${METHOD_ARRAY[@]}"; do
                    method="$(trim "$method_raw")"
                    for src in "${sources[@]}"; do
                        for mod_raw in "${MODALITY_ARRAY[@]}"; do
                            mod="$(trim "$mod_raw")"
                            cfg="configs/task1/${method}/${src}_to_${target}_${mod}.yaml"
                            if config_exists_or_handle "$cfg"; then
                                CONFIGS+=("$cfg")
                            fi
                        done
                    done
                done
                ;;

            2)
                # Task2: 0.1T -> Higher
                # targets: 1.5T / 3T / 5T / 7T
                local source="0.1T"
                local targets=("1.5T" "3T" "5T" "7T")

                for method_raw in "${METHOD_ARRAY[@]}"; do
                    method="$(trim "$method_raw")"
                    for tgt in "${targets[@]}"; do
                        for mod_raw in "${MODALITY_ARRAY[@]}"; do
                            mod="$(trim "$mod_raw")"
                            cfg="configs/task2/${method}/${source}_to_${tgt}_${mod}.yaml"
                            if config_exists_or_handle "$cfg"; then
                                CONFIGS+=("$cfg")
                            fi
                        done
                    done
                done
                ;;

            3)
                # Task3: StarGAN v2 Any -> Any
                # 默认按 modality-specific config 跑。
                for mod_raw in "${MODALITY_ARRAY[@]}"; do
                    mod="$(trim "$mod_raw")"
                    cfg="configs/task3/stargan/any_to_any_${mod}.yaml"
                    if config_exists_or_handle "$cfg"; then
                        CONFIGS+=("$cfg")
                    fi
                done
                ;;

            *)
                echo "[ERROR] Unknown task: $task"
                exit 1
                ;;
        esac
    done
}

run_one() {
    local cfg="$1"
    local mode="$2"

    if should_skip_done "$cfg" "$mode"; then
        return 0
    fi

    local tag
    tag="$(sanitize_tag "$cfg" "$mode")"

    echo ""
    echo "============================================================"
    echo "START"
    echo "Config: ${cfg}"
    echo "Mode:   ${mode}"
    echo "GPUs:   ${GPUS}"
    echo "NPROC:  ${NPROC}"
    echo "Port:   ${MASTER_PORT}"
    echo "Log:    logs/${tag}.log"
    echo "============================================================"
    echo ""

    CMD=(
        env
        CUDA_DEVICE_ORDER=PCI_BUS_ID
        CUDA_VISIBLE_DEVICES="${GPUS}"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        torchrun
        --standalone
        --nnodes=1
        --nproc_per_node="${NPROC}"
        --master_port="${MASTER_PORT}"
        scripts/train.py
        --config "${cfg}"
        --mode "${mode}"
        --dist
    )

    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    echo ""

    if [[ "$DRY_RUN" -eq 0 ]]; then
        "${CMD[@]}" 2>&1 | tee "logs/${tag}.log"
    fi

    echo ""
    echo "============================================================"
    echo "DONE"
    echo "Config: ${cfg}"
    echo "Mode:   ${mode}"
    echo "============================================================"
    echo ""
}

# 读取 .env，主要为了 --skip-done 时定位 OUTPUT_DIR。
if [[ -f "../.env" ]]; then
    set -a
    source "../.env"
    set +a
fi

build_config_list

echo ""
echo "============================================================"
echo "Full config training plan"
echo "Tasks:       ${TASKS}"
echo "Methods:     ${METHODS}"
echo "Modalities:  ${MODALITIES}"
echo "Modes:       ${MODES}"
echo "GPUs:        ${GPUS}"
echo "NPROC:       ${NPROC}"
echo "Dry run:     ${DRY_RUN}"
echo "Skip done:   ${SKIP_DONE}"
echo "Num configs: ${#CONFIGS[@]}"
echo "============================================================"
echo ""

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
    echo "[ERROR] No configs found."
    exit 1
fi

echo "Config list:"
for cfg in "${CONFIGS[@]}"; do
    echo "  - $cfg"
done
echo ""

case "$MODES" in
    retro)
        echo "Stage: retro_scratch"
        for cfg in "${CONFIGS[@]}"; do
            run_one "$cfg" "retro_scratch"
        done
        ;;

    pro)
        echo "Stage: pro_pretrained"
        for cfg in "${CONFIGS[@]}"; do
            run_one "$cfg" "pro_pretrained"
        done
        ;;

    full)
        echo "Stage 1: retro_scratch for all selected configs"
        for cfg in "${CONFIGS[@]}"; do
            run_one "$cfg" "retro_scratch"
        done

        echo "Stage 2: pro_pretrained for all selected configs"
        for cfg in "${CONFIGS[@]}"; do
            run_one "$cfg" "pro_pretrained"
        done
        ;;
esac

echo ""
echo "All selected config training jobs finished."
