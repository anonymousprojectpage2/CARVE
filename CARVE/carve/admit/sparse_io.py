"""
Sparse storage I/O for Method A/A+ merging.

  - τ_merge is shared across all tasks (saved once in shared/ as bf16).
  - Each task stores only (binary mask packed as uint8 bits, β dict).
  - Shapes/dtypes are recovered from base_sd at load time.

Storage scaling (7B OpenVLA, N tasks):
  shared    ≈ 14 GB (base; not stored here) + 14 GB (τ_merge bf16)
  per-task  ≈ 7B bits ÷ 8 ≈ 875 MB (mask) + <1 KB (β)
"""

import json
import os
from typing import Dict, Iterable, Tuple, List

import numpy as np
import torch
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------
# Bit packing (numpy.packbits uses bitorder='big' by default; we preserve it)
# ---------------------------------------------------------------------
def pack_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    """Bool tensor -> 1D uint8 tensor of ceil(N/8) bytes."""
    arr = mask.detach().cpu().numpy().astype(np.uint8).flatten()
    packed = np.packbits(arr)  # big-endian bit order
    return torch.from_numpy(packed.copy())  # contiguous copy


def unpack_mask_tensor(
    packed: torch.Tensor, total_elems: int, shape: torch.Size
) -> torch.Tensor:
    """1D uint8 -> bool tensor of given shape. Strips the padding bits."""
    packed_np = packed.detach().cpu().numpy().astype(np.uint8)
    unpacked = np.unpackbits(packed_np)[:total_elems]
    return torch.from_numpy(unpacked.astype(np.bool_).copy()).view(shape)


# ---------------------------------------------------------------------
# Shared save/load (τ_merge + metadata)
# ---------------------------------------------------------------------
def save_shared(
    shared_dir: str,
    tau_merge: Dict[str, torch.Tensor],
    identical_keys: Iterable[str],
    merge_config: dict,
    shared_dataset_stats: dict,
    tau_dtype: torch.dtype = torch.bfloat16,
):
    os.makedirs(shared_dir, exist_ok=True)

    tau_to_save = {k: v.to(tau_dtype).contiguous() for k, v in tau_merge.items()}
    save_file(tau_to_save, os.path.join(shared_dir, "tau_merge.safetensors"))

    with open(os.path.join(shared_dir, "identical_keys.json"), "w") as f:
        json.dump(sorted(list(identical_keys)), f, indent=2)
    with open(os.path.join(shared_dir, "merge_config.json"), "w") as f:
        json.dump(merge_config, f, indent=2)
    with open(os.path.join(shared_dir, "dataset_stats.json"), "w") as f:
        json.dump(_json_safe(shared_dataset_stats), f, indent=2)


def load_shared(
    shared_dir: str,
) -> Tuple[Dict[str, torch.Tensor], set, dict, dict]:
    """Returns (tau_merge_fp32, identical_keys, merge_config, dataset_stats)."""
    tau_bf16 = load_file(os.path.join(shared_dir, "tau_merge.safetensors"))
    tau_merge = {k: v.float() for k, v in tau_bf16.items()}  # fp32 for arithmetic

    with open(os.path.join(shared_dir, "identical_keys.json")) as f:
        identical_keys = set(json.load(f))
    with open(os.path.join(shared_dir, "merge_config.json")) as f:
        merge_config = json.load(f)
    with open(os.path.join(shared_dir, "dataset_stats.json")) as f:
        shared_dataset_stats = json.load(f)
    return tau_merge, identical_keys, merge_config, shared_dataset_stats


# ---------------------------------------------------------------------
# Per-task save/load (packed mask + β + stats)
# ---------------------------------------------------------------------
def save_task_sparse(
    task_dir: str,
    mask_dict: Dict[str, torch.Tensor],  # {key: bool tensor, original param shape}
    beta_by_group: dict,
    stats: dict,
):
    os.makedirs(task_dir, exist_ok=True)
    packed_dict = {k: pack_mask_tensor(m) for k, m in mask_dict.items()}
    save_file(packed_dict, os.path.join(task_dir, "mask_packed.safetensors"))

    with open(os.path.join(task_dir, "beta.json"), "w") as f:
        json.dump(_json_safe(beta_by_group), f, indent=2)
    with open(os.path.join(task_dir, "stats.json"), "w") as f:
        json.dump(_json_safe(stats), f, indent=2)


def load_task_sparse(
    task_dir: str, base_sd: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Returns (mask_dict_bool, beta_by_group). Shape is recovered from base_sd."""
    packed = load_file(os.path.join(task_dir, "mask_packed.safetensors"))
    mask_dict: Dict[str, torch.Tensor] = {}
    for k, p in packed.items():
        if k not in base_sd:
            raise KeyError(f"Key '{k}' in saved mask but not in base state dict.")
        mask_dict[k] = unpack_mask_tensor(p, base_sd[k].numel(), base_sd[k].shape)
    with open(os.path.join(task_dir, "beta.json")) as f:
        beta_by_group = json.load(f)
    return mask_dict, beta_by_group


# ---------------------------------------------------------------------
# Storage reporting
# ---------------------------------------------------------------------
def report_storage(save_root: str, task_names: List[str]) -> dict:
    def _dir_bytes(d):
        if not os.path.isdir(d):
            return 0
        total = 0
        for root, _, files in os.walk(d):
            for fn in files:
                total += os.path.getsize(os.path.join(root, fn))
        return total

    shared_bytes = _dir_bytes(os.path.join(save_root, "shared"))
    per_task_bytes = {
        t: _dir_bytes(os.path.join(save_root, t)) for t in task_names
    }
    total = shared_bytes + sum(per_task_bytes.values())
    return {
        "shared_bytes": shared_bytes,
        "per_task_bytes": per_task_bytes,
        "total_bytes": total,
        "shared_gb": shared_bytes / 1024 ** 3,
        "per_task_gb": {t: b / 1024 ** 3 for t, b in per_task_bytes.items()},
        "total_gb": total / 1024 ** 3,
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    try:
        return float(o)
    except Exception:
        return str(o)