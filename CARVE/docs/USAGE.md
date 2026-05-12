# CARVE — Usage Guide

This guide walks through the end-to-end pipeline, from a clean OpenVLA
checkout to a routing-eval log file with success rates.

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| OpenVLA repo | latest | installed in editable mode |
| LIBERO | latest | benchmark + experts |
| PyTorch | ≥ 2.2 | with CUDA |
| HuggingFace cache | populated | pretrained + 4 LIBERO experts |
| GPU | ≥ 40 GB | for admit; eval needs ≥ 24 GB |

```bash
# 1. Install CARVE
pip install -e .
pip install -r requirements.txt

# 2. Point at OpenVLA
export OPENVLA_REPO=/path/to/openvla
export HF_HOME=/path/to/hf_cache

# 3. Link the package so OpenVLA scripts can import it
bash setup.sh
```

After `setup.sh` you can verify the import:

```bash
cd $OPENVLA_REPO
python -c "from carve.admit.merge import _self_test; _self_test()" || true
```

## 1. Admit — build a bundle

A single rank:

```bash
GPU=0 \
RANK=64 \
BUNDLE=./carve_T4_r64 \
bash scripts/admit.sh
```

A bundle is created at `$BUNDLE` containing the shared core, per-task
residuals, and per-step admit logs. The full rank ablation:

```bash
GPU=0 BUNDLE_ROOT=./carve_T4 bash scripts/admit_rank_ablation.sh
# Produces ./carve_T4_r16, ./carve_T4_r32, ./carve_T4_r64, ./carve_T4_r128
```

Admit time on a single 96-GB GPU for T=4 LIBERO is roughly 15–25 minutes per
rank, dominated by per-key randomized SVDs.

## 2. Evaluate — oracle overlay

Each task uses its own admitted residual. 4 GPUs in parallel:

```bash
BUNDLE=./carve_T4_r64 \
TRIALS=10 \
GPUS=0,1,2,3 \
bash scripts/eval_oracle.sh
```

Logs are written to `$BUNDLE/eval_oracle/` with one file per LIBERO suite.
Each log ends with a summary block:

```
Final results:
  Total episodes: 50
  Total successes: 43
  Overall success rate: 0.8600 (86.0%)
```

## 3. Evaluate — routing overlay

The routing eval needs per-task keys; build them once per bundle:

```bash
GPU=0 BUNDLE=./carve_T4_r64 bash scripts/build_keys.sh
```

This writes `$BUNDLE/routing_keys/`. Now run routing eval for any K:

```bash
for K in 1 2 4; do
  BUNDLE=./carve_T4_r64 K=$K \
  TRIALS=10 GPUS=0,1,2,3 \
  bash scripts/eval_routing.sh
done
```

Logs are at `$BUNDLE/eval_routing_K${K}/`. The success-rate format is the
same as the oracle eval.

## 4. Programmatic use

For one-off experiments, the package can also be driven from Python — see
`examples/quickstart.py`.

## Troubleshooting

- **`ModuleNotFoundError: carve`** when running a script: confirm that
  `setup.sh` linked `carve/` into `$OPENVLA_REPO/carve_pkg` and that
  `PYTHONPATH` includes `$OPENVLA_REPO/carve_pkg`.
- **CUDA OOM during admit**: try `--min_factorise_numel 8192` (or larger)
  so more keys are factorised rather than stored as raw vectors. Or admit on
  a GPU with more memory.
- **Routing eval reports a "missing routing keys" error**: run
  `scripts/build_keys.sh` first.
