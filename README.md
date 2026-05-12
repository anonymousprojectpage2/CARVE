# CARVE: Skill-Preserving Continual Merging of Vision-Language-Action Experts

**Anonymous submission to NeurIPS 2026**

> CARVE (**C**ontinual **A**daptive **R**esidual-preserving **V**LA **E**xpert merging) is a continual post-hoc VLA expert merging framework that maintains a skill-preserving merge state consisting of a shared global core and compact skill-local spectral residuals.

[[Project Page]](https://anonymousprojectpage2.github.io/CARVE/)

---

## Overview

CARVE integrates sequentially arriving task-specific VLA experts without joint retraining, past robot data, full expert retention, or architecture redesign. For each incoming expert:
1. Concordant update directions are accumulated into a **shared global core**
2. Non-shared updates are preserved as **compact low-rank spectral residuals**
3. At inference time, each skill is instantiated from the base model + final global core + its residual

---

## Quick Start

### 0. Prerequisites

- Working OpenVLA / VLA-Adapter / MergeVLA repository
- LIBERO benchmark installed and importable
- HuggingFace cache populated with base and expert models (see [Expert Models](#expert-models))
- A GPU with ≥40 GB memory for merging, ≥24 GB for evaluation

### 1. Install

```bash
git clone https://github.com/anonymousprojectpage2/CARVE.git
cd CARVE
conda create -n carve python=3.10 -y
conda activate carve
pip install -r requirements.txt

# Install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..
```

### 2. Admit (build a merge state)

T=4 LIBERO experts at rank 64 (the main paper setting):

```bash
bash bash_scripts/continual_merge_libero.sh OpenVLA
```

### 3. Evaluate

```bash
CKPT=./kj/continual_merged_models/OpenVLA/LIBERO/order_spatial_object_goal_10/step_4
# See Evaluation section for per-backbone instructions
```

---

## Repository Structure

```
CARVE/
├── bash_scripts/               # Shell scripts for training, merging, and evaluation
│   ├── continual_merge_libero.sh        # Continual merging for LIBERO
│   ├── continual_merge_libero_plus.sh   # Continual merging for LIBERO-Plus
│   └── finetune_libero_plus.sh          # Fine-tuning on LIBERO-Plus
├── experiments/robot/          # Evaluation scripts
│   └── libero/
│       └── run_libero_eval.py           # LIBERO evaluation (MergeVLA backbone)
├── model_merging/              # Core CARVE merging code
│   ├── continual_mergy.py               # Continual merging for OpenVLA / VLA-Adapter
│   └── continual_mergy_MergeVLA.py      # Continual merging for MergeVLA
├── pretrained_models/configs/  # Model configuration files
├── prismatic/                  # Prismatic VLM utilities
├── vla-scripts/                # Fine-tuning scripts
│   ├── finetune.py                      # LIBERO fine-tuning
│   └── finetune_libero_plus.py          # LIBERO-Plus fine-tuning
└── README.md
```

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/anonymousprojectpage2/CARVE.git
cd CARVE

# Create conda environment
conda create -n carve python=3.10 -y
conda activate carve

# Install dependencies
pip install -r requirements.txt

# Install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..
```

---

## Expert Models

CARVE uses publicly released task-specific experts. Download them from HuggingFace:

| Model | Suite | HuggingFace ID |
|-------|-------|----------------|
| OpenVLA | Spatial | `openvla/openvla-7b-finetuned-libero-spatial` |
| OpenVLA | Object | `openvla/openvla-7b-finetuned-libero-object` |
| OpenVLA | Goal | `openvla/openvla-7b-finetuned-libero-goal` |
| OpenVLA | Long | `openvla/openvla-7b-finetuned-libero-10` |
| VLA-Adapter-Pro | Spatial | `VLA-Adapter/LIBERO-Spatial-Pro` |
| VLA-Adapter-Pro | Object | `VLA-Adapter/LIBERO-Object-Pro` |
| VLA-Adapter-Pro | Goal | `VLA-Adapter/LIBERO-Goal-Pro` |
| VLA-Adapter-Pro | Long | `VLA-Adapter/LIBERO-Long-Pro` |
| MergeVLA | Spatial | Available via [MergeVLA repository](https://github.com/MergeVLA/MergeVLA) |
| MergeVLA | Object | Available via [MergeVLA repository](https://github.com/MergeVLA/MergeVLA) |
| MergeVLA | Goal | Available via [MergeVLA repository](https://github.com/MergeVLA/MergeVLA) |
| MergeVLA | Long | Available via [MergeVLA repository](https://github.com/MergeVLA/MergeVLA) |

Pretrained base models:

| Model | Path |
|-------|------|
| OpenVLA base | `openvla/openvla-7b` |
| VLA-Adapter base | `VLA-Adapter/pretrained_models/vla_config` |
| MergeVLA base | Available via MergeVLA repository |

---

## LIBERO-Plus Expert Fine-tuning

For LIBERO-Plus experiments, we fine-tune LIBERO experts further on LIBERO-Plus data using LoRA (rank=64, lr=2e-4, batch size=8, 10k steps).

**Step 1: Download LIBERO-Plus dataset**

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Sylvest/libero_plus_data_4suite',
    repo_type='dataset',
    local_dir='data/libero_plus_rlds',
    allow_patterns='rlds/*',
)
"
```

**Step 2: Run fine-tuning**

Edit `bash_scripts/finetune_libero_plus.sh` to set:
- `MODEL_TYPE`: `MergeVLA` / `OpenVLA` / `VLAAdapter`

```bash
bash bash_scripts/finetune_libero_plus.sh
```

---

## Continual Merging

### CARVE on LIBERO

Pass the model type as an argument:

```bash
bash bash_scripts/continual_merge_libero.sh MergeVLA
bash bash_scripts/continual_merge_libero.sh OpenVLA
bash bash_scripts/continual_merge_libero.sh VLAAdapter
```

### CARVE on LIBERO-Plus

```bash
bash bash_scripts/continual_merge_libero_plus.sh MergeVLA
bash bash_scripts/continual_merge_libero_plus.sh OpenVLA
bash bash_scripts/continual_merge_libero_plus.sh VLAAdapter
```

### Merge State

After merging, checkpoints are saved under `SAVE_DIR/order_{task_order}/`:

```
order_spatial_object_goal_10/
├── step_1/              # After admitting spatial expert
├── step_2/              # After admitting object expert
├── step_3/              # After admitting goal expert
├── step_4/              # After admitting long expert
└── continual_state.pt   # Resumable merge state
```

---

## Storage Cost (OpenVLA-7B, T=4 LIBERO, rank=64)

| Component | Size |
|-----------|------|
| Pretrained OpenVLA-7B (frozen, shared) | 7.54 B |
| Shared core delta (τ_core) | 7.54 B |
| Per-task residuals (T=4 combined) | 0.87 B |
| **Effective state for 4 tasks** | **15.95 B** |
| Naive (4 independent copies) | 30.16 B |
| **Saving vs. naive** | **47%** |

---

## Evaluation

Each VLA backbone has its own evaluation codebase. We recommend cloning the corresponding repository and running evaluation from within that environment.

---

### OpenVLA

```bash
# Clone and set up OpenVLA
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

CKPT=/path/to/CARVE/merged/OpenVLA/order_spatial_object_goal_10/step_4
for task in libero_spatial libero_object libero_goal libero_10; do
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint $CKPT \
        --task_suite_name $task \
        --num_trials_per_task 50 \
        --center_crop True
done
```

---

### VLA-Adapter

```bash
# Clone and set up VLA-Adapter
git clone https://github.com/VLA-Adapter/VLA-Adapter.git
cd VLA-Adapter
pip install -e .

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

CKPT=/path/to/CARVE/merged/VLAAdapter/order_spatial_object_goal_10/step_4
for task in libero_spatial libero_object libero_goal libero_10; do
    python experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint $CKPT \
        --task_suite_name $task \
        --num_trials_per_task 50
done
```

---

### MergeVLA

```bash
# Clone and set up MergeVLA
git clone https://github.com/MergeVLA/MergeVLA.git
cd MergeVLA
pip install -e .

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

CKPT=/path/to/CARVE/merged/MergeVLA/order_spatial_object_goal_10/step_4
for task in libero_spatial libero_object libero_goal libero_10; do
    python experiments/robot/libero/run_libero_eval.py \
        --num_images_in_input 2 \
        --pretrained_checkpoint $CKPT \
        --task_suite_name $task \
        --load_moe True \
        --pretrained_vlm_checkpoint /path/to/MergeVLA/pretrained_models/Pretrained-VLM \
        --k_gate 8 \
        --action_head_layer_num 1 \
        --num_trials_per_task 50
done
```


## License

Apache 2.0. See `LICENSE`.
