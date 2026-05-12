#!/bin/bash
# CARVE — rank ablation. Admits T=4 LIBERO at ranks {16, 32, 64, 128}.
# Usage:
#   GPU=0 bash scripts/admit_rank_ablation.sh

set -e
: "${GPU:=0}"
: "${RANKS:="16 32 64 128"}"
: "${BUNDLE_ROOT:=./carve_T4}"

for R in $RANKS; do
    BUNDLE="${BUNDLE_ROOT}_r${R}"
    if [ -d "$BUNDLE/libero_10" ] && [ -f "$BUNDLE/libero_10/stats.json" ]; then
        echo "[r=$R] already done, skip"
        continue
    fi
    echo "[r=$R] admit -> $BUNDLE  ($(date))"
    GPU=$GPU RANK=$R BUNDLE=$BUNDLE bash "$(dirname "$0")/admit.sh"
done

echo "[rank ablation] all done."
