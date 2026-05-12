#!/bin/bash
# CARVE — single-rank admit (T=4 LIBERO tasks).
# Usage:
#   GPU=0 RANK=64 BUNDLE=./carve_T4_r64 bash scripts/admit.sh

set -e
: "${OPENVLA_REPO:?Please set OPENVLA_REPO=/path/to/openvla}"
: "${BUNDLE:?Please set BUNDLE=./carve_T4_r<rank>}"
: "${RANK:=64}"
: "${GPU:=0}"
: "${HF_HOME:=$HOME/.cache/huggingface}"

cd "$OPENVLA_REPO"
export PYTHONPATH="$OPENVLA_REPO:$PYTHONPATH"
export HF_HOME

TASKS=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(
  openvla/openvla-7b-finetuned-libero-spatial
  openvla/openvla-7b-finetuned-libero-object
  openvla/openvla-7b-finetuned-libero-goal
  openvla/openvla-7b-finetuned-libero-10
)

mkdir -p "$BUNDLE/sequence_logs"
echo "[CARVE admit] bundle=$BUNDLE rank=$RANK gpu=$GPU"

for i in "${!TASKS[@]}"; do
    step=$((i + 1))
    task="${TASKS[$i]}"
    ckpt="${CKPTS[$i]}"
    log="$BUNDLE/sequence_logs/step_${step}_admit_${task}.log"

    if [ -d "$BUNDLE/$task" ] && [ -f "$BUNDLE/$task/stats.json" ]; then
        echo "[step $step] skip $task (already admitted)"
        continue
    fi

    echo "[step $step/${#TASKS[@]}] admit $task ($(date +%H:%M:%S))"

    CUDA_VISIBLE_DEVICES=$GPU \
    python -m carve.admit.merge \
        --bundle_root "$BUNDLE" \
        --task_name "$task" \
        --task_ckpt "$ckpt" \
        --base_ckpt openvla/openvla-7b \
        --gamma 1.0 \
        --alpha_mode inv_sqrt \
        --scope per_block \
        --rank_max "$RANK" \
        --rank_adaptive_threshold 0.0 \
        --min_factorise_numel 4096 \
        --use_beta 1 \
        > "$log" 2>&1

    echo "  done -> $log"
done

echo "[CARVE admit] all done: $BUNDLE"
