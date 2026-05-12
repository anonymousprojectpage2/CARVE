#!/bin/bash
# CARVE — build per-task routing keys (one-time, after admit).
# Usage:
#   GPU=0 BUNDLE=./carve_T4_r64 bash scripts/build_keys.sh

set -e
: "${OPENVLA_REPO:?Please set OPENVLA_REPO}"
: "${BUNDLE:?Please set BUNDLE=./carve_T4_r<rank>}"
: "${GPU:=0}"

cd "$OPENVLA_REPO"
export PYTHONPATH="$OPENVLA_REPO:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=$GPU \
python -m carve.routing.keys \
    --bundle_root "$BUNDLE"

echo "[CARVE routing keys] built -> $BUNDLE/routing_keys/"
