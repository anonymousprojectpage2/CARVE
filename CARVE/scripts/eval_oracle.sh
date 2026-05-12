#!/bin/bash
# CARVE — oracle eval: each task uses its own admitted residual.
# Usage:
#   BUNDLE=./carve_T4_r64 TRIALS=10 GPUS=0,1,2,3 bash scripts/eval_oracle.sh

set -e
: "${OPENVLA_REPO:?Please set OPENVLA_REPO}"
: "${BUNDLE:?Please set BUNDLE=./carve_T4_r<rank>}"
: "${TRIALS:=10}"
: "${GPUS:="0,1,2,3"}"
: "${HF_HOME:=$HOME/.cache/huggingface}"

cd "$OPENVLA_REPO"
export PYTHONPATH="$OPENVLA_REPO:$PYTHONPATH"
export HF_HOME

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
TASKS=(libero_spatial libero_object libero_goal libero_10)

OUTDIR="$BUNDLE/eval_oracle"
mkdir -p "$OUTDIR"

pids=()
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    log="$OUTDIR/${task}.log"
    echo "[GPU $gpu] eval $task -> $log"
    CUDA_VISIBLE_DEVICES=$gpu \
    python -m carve.eval.oracle \
        --pretrained_checkpoint "$BUNDLE/$task" \
        --task_suite_name "$task" \
        --center_crop True \
        --num_trials_per_task "$TRIALS" \
        > "$log" 2>&1 &
    pids+=("$!")
done

for p in "${pids[@]}"; do wait "$p"; done
echo "[oracle eval] done. logs in $OUTDIR"
