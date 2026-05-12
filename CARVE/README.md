# CARVE: Continual Admission with Residual Vector Experts

CARVE is a streaming approach to continually merging task-specific fine-tunes
of a Vision–Language–Action (VLA) model. It maintains a single shared core
delta on top of the frozen pretrained backbone, plus a small per-task residual
for each admitted task. Two inference variants are provided:

- **Oracle overlay** — the task identity is known at inference; the model uses
  the residual that belongs to that task.
- **Routing overlay** — the task identity is *not* known; a lightweight
  per-task routing key picks the top-K residuals and mixes them at inference
  time.

This repository contains the OpenVLA-7B implementation used in our paper.

## Repository layout

```
CARVE/
├── README.md                          # this file
├── LICENSE
├── requirements.txt
├── setup.py                           # pip install -e .
├── setup.sh                           # symlinks carve/ into your OpenVLA repo
│
├── carve/                             # Python package
│   ├── __init__.py
│   ├── admit/
│   │   ├── merge.py                   # streaming admit (sign-protected EMA + SVD)
│   │   ├── io.py                      # bundle I/O
│   │   └── utils.py                   # common utilities
│   ├── eval/
│   │   ├── oracle.py                  # oracle evaluation entry point
│   │   └── overlay_oracle.py          # oracle overlay loader
│   └── routing/
│       ├── keys.py                    # build per-task routing keys
│       ├── keys_io.py                 # routing-key I/O
│       ├── router.py                  # K-shot router core
│       ├── overlay.py                 # routing overlay loader
│       ├── eval.py                    # routing evaluation entry point
│       ├── probe.py                   # diagnostic probing
│       ├── text_probe.py              # text-conditioned routing variant
│       └── score_patch.py             # routing score utilities
│
├── scripts/
│   ├── admit.sh                       # admit T=4 tasks at a chosen rank
│   ├── admit_rank_ablation.sh         # admit at multiple ranks
│   ├── eval_oracle.sh                 # oracle eval
│   ├── build_keys.sh                  # build routing keys (one-time)
│   └── eval_routing.sh                # routing eval at top-K
│
├── configs/
│   └── default.yaml
│
├── docs/
│   ├── ALGORITHM.md                   # method description
│   ├── USAGE.md                       # end-to-end how-to
│   └── REPRODUCING.md                 # reproducing paper numbers
│
└── examples/
    └── quickstart.py
```

## Quick start

### 0. Prerequisites

- Working OpenVLA repository, installable with `pip install -e .`
- LIBERO benchmark installed and importable
- HuggingFace cache populated with:
  - `openvla/openvla-7b` (pretrained base)
  - `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}` (experts)
- A GPU with ≥40 GB memory for admit, ≥24 GB for evaluation

### 1. Install

```bash
pip install -e .
pip install -r requirements.txt

export OPENVLA_REPO=/path/to/openvla
export HF_HOME=/path/to/hf_cache
bash setup.sh                # symlinks carve/ -> $OPENVLA_REPO/carve_pkg
```

### 2. Admit (build a bundle)

T=4 LIBERO at rank 64 (the main paper setting):

```bash
GPU=0 \
RANK=64 \
BUNDLE=./carve_T4_r64 \
bash scripts/admit.sh
```

Or run the full rank ablation in one go:

```bash
GPU=0 bash scripts/admit_rank_ablation.sh
```

A bundle is laid out as:

```
carve_T4_r64/
├── shared/
│   └── tau_core.safetensors           # shared core delta (~ pretrained-size, bf16)
├── libero_spatial/
│   ├── residual.safetensors           # per-task LR factors + bf16 vectors
│   └── stats.json                     # admit diagnostics
├── libero_object/...
├── libero_goal/...
├── libero_10/...
└── sequence_logs/                     # per-step admit logs
```

### 3a. Evaluate — Oracle overlay

```bash
BUNDLE=./carve_T4_r64 \
TRIALS=10 \
GPUS=0,1,2,3 \
bash scripts/eval_oracle.sh
```

Logs are written to `$BUNDLE/eval_oracle/`.

### 3b. Evaluate — Routing overlay

First build routing keys once per bundle:

```bash
GPU=0 BUNDLE=./carve_T4_r64 bash scripts/build_keys.sh
```

Then run routing eval at any K:

```bash
for K in 1 2 4; do
  BUNDLE=./carve_T4_r64 K=$K \
  TRIALS=10 GPUS=0,1,2,3 \
  bash scripts/eval_routing.sh
done
```

Logs are written to `$BUNDLE/eval_routing_K{K}/`.

## Storage cost (OpenVLA-7B, T=4 LIBERO, rank=64)

| Component | Size |
|---|---|
| Pretrained OpenVLA-7B (frozen, shared) | 7.54 B |
| Shared core delta (τ_core) | 7.54 B |
| Per-task residuals (T=4 combined) | 0.87 B |
| **Effective state for 4 tasks** | **15.95 B** |
| Naive (4 independent copies) | 30.16 B |
| **Saving vs. naive** | **47%** |

See `docs/ALGORITHM.md` for the full derivation and `docs/REPRODUCING.md` for
how the numbers map to the tables in the paper.

## Citation

```bibtex
@inproceedings{carve2026,
  title  = {CARVE: Continual Admission with Residual Vector Experts for VLA Models},
  author = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year   = {2026}
}
```

## License

Apache 2.0. See `LICENSE`.
