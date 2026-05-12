"""In-memory overlay for CMM LR-Residual bundles."""

from __future__ import annotations

import gc
import json
import os
import sys
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file as load_safetensors

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_MERGE_SRC = os.environ.get("MERGEVLA_SRC", _SELF_DIR)
for _p in (_SELF_DIR, _MERGE_SRC):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from carve.admit.utils import classify_key  # noqa: E402


_CORE = "tau_core.safetensors"
_RESID = "residual.safetensors"
_RANKS = "ranks.json"


def detect_lr_bundle(path: str) -> Optional[Tuple[str, str]]:
    """Return (bundle_root, task_name) if `path` points into a CMM-LR bundle."""
    if not isinstance(path, str) or not os.path.isdir(path):
        return None
    task_dir = os.path.abspath(path.rstrip("/"))
    bundle_root = os.path.dirname(task_dir)
    if not os.path.exists(os.path.join(task_dir, _RESID)):
        return None
    if not os.path.exists(os.path.join(bundle_root, "shared", _CORE)):
        return None
    cfg_path = os.path.join(bundle_root, "shared", "merge_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("method") not in {"CMM-LR"}:
            return None
    except Exception:
        return None
    return bundle_root, os.path.basename(task_dir)


def load_lr_merge_config(bundle_root: str) -> dict:
    with open(os.path.join(bundle_root, "shared", "merge_config.json")) as f:
        return json.load(f)


def load_lr_dataset_stats(bundle_root: str) -> dict:
    path = os.path.join(bundle_root, "shared", "dataset_stats.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _group_of(key: str) -> str:
    info = classify_key(key)
    if info["layer_idx"] >= 0:
        return f"llm_block_{info['layer_idx']:02d}"
    return f"non_llm_{info['module']}"


def _resolve_beta(beta_by_group: dict, scope: str, key: str) -> float:
    if scope == "global":
        return float(beta_by_group.get("global", 1.0))
    if scope == "per_block":
        return float(beta_by_group.get(_group_of(key), 1.0))
    if scope == "per_key":
        return float(beta_by_group.get(key, 1.0))
    return 1.0


@torch.no_grad()
def apply_lr_overlay_inplace_(
    model,
    bundle_root: str,
    task_name: str,
    merge_config: dict,
    *,
    verbose: bool = True,
    log_prefix: str = "[lr overlay]",
) -> dict:
    """Apply theta_m = theta_0 + beta_m * tau_core + residual_m.

    residual_m[k] is reconstructed from:
      - (U, V) -> U @ V.T reshaped to original shape, OR
      - vec    -> reshaped directly.
    """
    scope = merge_config.get("scope", "per_block")
    use_beta = bool(merge_config.get("use_beta", True))
    shared_path = os.path.join(bundle_root, "shared", _CORE)
    task_dir = os.path.join(bundle_root, task_name)

    if verbose:
        print(f"{log_prefix} reading shared/{_CORE} (bf16)")
    tau_core = load_safetensors(shared_path)

    if verbose:
        print(f"{log_prefix} reading {task_name}/{_RESID}")
    residual = load_safetensors(os.path.join(task_dir, _RESID))
    with open(os.path.join(task_dir, "beta.json")) as f:
        beta_by_group = json.load(f)

    sd = dict(model.state_dict())
    stat = {
        "applied_core_keys": 0,
        "applied_residual_lr_keys": 0,
        "applied_residual_vec_keys": 0,
        "skipped_missing": 0,
        "skipped_shape": 0,
        "total_positions": 0,
    }

    for k, core_bf in tau_core.items():
        p = sd.get(k)
        if p is None:
            stat["skipped_missing"] += 1
            continue
        if tuple(p.shape) != tuple(core_bf.shape):
            stat["skipped_shape"] += 1
            continue

        # Core contribution (applied to ALL positions; mask-less design).
        beta = _resolve_beta(beta_by_group, scope, k) if use_beta else 1.0
        delta = core_bf.float() * beta  # shape == p.shape

        # Residual contribution, if any.
        Uk = residual.get(f"{k}.U")
        Vk = residual.get(f"{k}.V")
        vec = residual.get(f"{k}.vec")
        shape_t = residual.get(f"{k}.shape")

        if Uk is not None and Vk is not None and shape_t is not None:
            orig_shape = tuple(int(x) for x in shape_t.tolist())
            lr = Uk.float() @ Vk.float().T  # (m, n')
            lr = lr.reshape(orig_shape)
            if tuple(lr.shape) == tuple(p.shape):
                delta = delta + lr
                stat["applied_residual_lr_keys"] += 1
            else:
                stat["skipped_shape"] += 1
            del lr
        elif vec is not None and shape_t is not None:
            orig_shape = tuple(int(x) for x in shape_t.tolist())
            v = vec.float().reshape(orig_shape)
            if tuple(v.shape) == tuple(p.shape):
                delta = delta + v
                stat["applied_residual_vec_keys"] += 1
            else:
                stat["skipped_shape"] += 1
            del v

        delta_dev = delta.to(device=p.device, dtype=torch.float32, non_blocking=True)
        p_fp32 = p.data.to(torch.float32)
        p_fp32.add_(delta_dev)
        p.data.copy_(p_fp32.to(p.dtype))

        stat["applied_core_keys"] += 1
        stat["total_positions"] += int(p.numel())
        del delta, delta_dev, p_fp32

    del tau_core, residual
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(
            f"{log_prefix} done  core_keys={stat['applied_core_keys']} "
            f"lr_keys={stat['applied_residual_lr_keys']} "
            f"vec_keys={stat['applied_residual_vec_keys']} "
            f"missing={stat['skipped_missing']} shape_skip={stat['skipped_shape']}"
        )
    return stat
