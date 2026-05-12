#!/bin/bash
# CARVE — routing eval (no oracle): K-shot router picks top-K residuals.
# Requires build_keys.sh to have run first.
# Usage:
#   BUNDLE=./carve_T4_r64 K=2 TRIALS=10 GPUS=0,1,2,3 bash scripts/eval_routing.sh

set -e
: "${OPENVLA_REPO:?Please set OPENVLA_REPO}"
: "${BUNDLE:?Please set BUNDLE=./carve_T4_r<rank>}"
: "${K:=2}"
: "${TRIALS:=10}"
: "${GPUS:="0,1,2,3"}"

cd "$OPENVLA_REPO"
export PYTHONPATH="$OPENVLA_REPO:$PYTHONPATH"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
TASKS=(libero_spatial libero_object libero_goal libero_10)

OUTDIR="$BUNDLE/eval_routing_K${K}"
mkdir -p "$OUTDIR"

pids=()
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    log="$OUTDIR/${task}.log"
    echo "[GPU $gpu] routing K=$K eval $task -> $log"
    CUDA_VISIBLE_DEVICES=$gpu \
    python -m carve.routing.eval \
        --bundle_root "$BUNDLE" \
        --task_suite_name "$task" \
        --k_route "$K" \
        --num_trials_per_task "$TRIALS" \
        > "$log" 2>&1 &
    pids+=("$!")
done

for p in "${pids[@]}"; do wait "$p"; done
echo "[routing eval] done. logs in $OUTDIR"
