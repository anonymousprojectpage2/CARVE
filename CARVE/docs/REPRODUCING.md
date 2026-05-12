# Reproducing Paper Numbers

This document maps each table/figure in the paper to a concrete reproduction
recipe. All numbers below are for OpenVLA-7B on LIBERO with T=4 admitted
tasks (spatial, object, goal, libero_10).

## Storage table (Table X in the paper)

```bash
GPU=0 RANK=64 BUNDLE=./carve_T4_r64 bash scripts/admit.sh
```

After admit, the bundle's on-disk size and component breakdown can be read
from `$BUNDLE/sequence_logs/step_*_admit_*.log`. The expected aggregate:

| Component | Size |
|---|---|
| Pretrained OpenVLA-7B (shared) | 7.54 B |
| Shared core delta τ_core | 7.54 B |
| Per-task residuals (T=4) | 0.87 B |
| **Total effective** | **15.95 B** |

## Rank ablation (Table Y)

```bash
GPU=0 BUNDLE_ROOT=./carve_T4 bash scripts/admit_rank_ablation.sh
```

This produces four bundles at ranks {16, 32, 64, 128}. Expected per-task
residual size (across T=4 tasks) and step-4 reconstruction error:

| Rank | Residual total | Recon error |
|---|---|---|
| 16  | 218 M  | 0.667 |
| 32  | 436 M  | 0.527 |
| 64  | 873 M  | 0.345 |
| 128 | 1746 M | 0.313 |

## Oracle eval (LIBERO, Table Z)

```bash
for R in 16 32 64 128; do
  BUNDLE=./carve_T4_r${R} \
  TRIALS=10 GPUS=0,1,2,3 \
  bash scripts/eval_oracle.sh
done
```

Per-task success rates appear at the end of each
`$BUNDLE/eval_oracle/${task}.log`.

## Routing eval (Table W)

```bash
for R in 16 32 64 128; do
  GPU=0 BUNDLE=./carve_T4_r${R} bash scripts/build_keys.sh
  for K in 1 2 4; do
    BUNDLE=./carve_T4_r${R} K=$K TRIALS=10 GPUS=0,1,2,3 \
    bash scripts/eval_routing.sh
  done
done
```

## Notes on randomness

- The randomized SVD used in admit is seeded internally by torch's default
  RNG; admit results should be deterministic given a fixed CUDA build.
- LIBERO evaluation includes stochastic environment resets; running with
  `--num_trials_per_task 10` averages over enough rollouts that the headline
  success rates are stable to within ±1%.
