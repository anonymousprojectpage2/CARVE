"""
ConcordRouter V2 — overlay loader.

Splits V1's `apply_lr_overlay_inplace_` into two stages:

  Stage A: apply_core_only_inplace_(model, bundle_root)
      model.weights += beta_default * tau_core
      where beta_default uses beta=1.0 globally (no task-specific scaling).
      => This is the SHARED PROBE MODEL theta_shared = theta_0 + C_T.

  Stage B: apply_task_residual_inplace_(model, bundle_root, task_name)
      model.weights += beta_m * tau_core_correction + residual_m
      where the correction switches beta from 1.0 (default) to beta_m
      (i.e. adds (beta_m - 1) * tau_core), then adds residual_m.
      => Together with stage A this gives the V1 task-instantiated model.

Reasoning: we want to do ONE shared forward (stage A only) for routing, then
patch in the selected task's contribution (stage B) for actual rollout. This
avoids a full base-model reload between probe and execution.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors


_CORE = "tau_core.safetensors"
_RESID = "residual.safetensors"


# -----------------------------------------------------------------------------
# Bundle config helpers (mirrors overlay_loader_lr.detect_lr_bundle)
# -----------------------------------------------------------------------------

def detect_v2_bundle(path: str):
    """Return (bundle_root, task_name) if path looks like a CMM-LR bundle dir."""
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
    return bundle_root, os.path.basename(task_dir)


def load_merge_config(bundle_root: str) -> dict:
    with open(os.path.join(bundle_root, "shared", "merge_config.json")) as f:
        return json.load(f)


def load_dataset_stats(bundle_root: str) -> dict:
    p = os.path.join(bundle_root, "shared", "dataset_stats.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _classify_layer_idx(key: str) -> int:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


def _group_of(key: str) -> str:
    li = _classify_layer_idx(key)
    if li >= 0:
        return f"llm_block_{li:02d}"
    # fallback module classification (rough; matches common_merge intent)
    if "vision" in key.lower():
        return "non_llm_vision"
    if "projector" in key.lower():
        return "non_llm_projector"
    if "embed" in key.lower():
        return "non_llm_embed"
    if "lm_head" in key.lower():
        return "non_llm_lm_head"
    return "non_llm_other"


def _resolve_beta(beta_by_group: dict, scope: str, key: str) -> float:
    if scope == "global":
        return float(beta_by_group.get("global", 1.0))
    if scope == "per_block":
        return float(beta_by_group.get(_group_of(key), 1.0))
    if scope == "per_key":
        return float(beta_by_group.get(key, 1.0))
    return 1.0


# -----------------------------------------------------------------------------
# Stage A: apply core only (shared probe model)
# -----------------------------------------------------------------------------

@torch.no_grad()
def apply_core_only_inplace_(
    model,
    bundle_root: str,
    *,
    verbose: bool = True,
    log_prefix: str = "[router-overlay]",
) -> dict:
    """theta <- theta_0 + tau_core   (no task residual, no beta scaling)."""
    shared_path = os.path.join(bundle_root, "shared", _CORE)
    if verbose:
        print(f"{log_prefix} STAGE A: applying tau_core only (probe model)")
        print(f"{log_prefix}   reading {shared_path}")
    tau_core = load_safetensors(shared_path)

    sd = dict(model.state_dict())
    stat = {"applied_keys": 0, "skipped_missing": 0, "skipped_shape": 0}
    for k, core_bf in tau_core.items():
        p = sd.get(k)
        if p is None:
            stat["skipped_missing"] += 1
            continue
        if tuple(p.shape) != tuple(core_bf.shape):
            stat["skipped_shape"] += 1
            continue
        delta = core_bf.to(device=p.device, dtype=torch.float32)
        p.data.add_(delta.to(dtype=p.dtype))
        stat["applied_keys"] += 1

    if verbose:
        print(f"{log_prefix}   applied={stat['applied_keys']} "
              f"missing={stat['skipped_missing']} shape_skip={stat['skipped_shape']}")

    del tau_core
    gc.collect()
    return stat


# -----------------------------------------------------------------------------
# Stage B: apply task residual + beta correction (probe -> exec)
# -----------------------------------------------------------------------------

@torch.no_grad()
def apply_task_correction_inplace_(
    model,
    bundle_root: str,
    task_name: str,
    merge_config: dict,
    *,
    verbose: bool = True,
    log_prefix: str = "[router-overlay]",
) -> dict:
    """Convert probe model (theta_0 + tau_core) into V1 task-instantiated model.

    delta_to_apply = (beta_m - 1) * tau_core + residual_m
    => after this call: theta = theta_0 + beta_m * tau_core + residual_m
    """
    scope = merge_config.get("scope", "per_block")
    use_beta = bool(merge_config.get("use_beta", True))

    shared_path = os.path.join(bundle_root, "shared", _CORE)
    task_dir = os.path.join(bundle_root, task_name)
    resid_path = os.path.join(task_dir, _RESID)

    if verbose:
        print(f"{log_prefix} STAGE B: applying task correction for {task_name}")

    tau_core = load_safetensors(shared_path)
    residual = load_safetensors(resid_path)
    with open(os.path.join(task_dir, "beta.json")) as f:
        beta_by_group = json.load(f)

    sd = dict(model.state_dict())
    stat = {
        "applied_core_correction": 0,
        "applied_residual_lr": 0,
        "applied_residual_vec": 0,
        "skipped_missing": 0,
        "skipped_shape": 0,
    }

    for k, core_bf in tau_core.items():
        p = sd.get(k)
        if p is None:
            stat["skipped_missing"] += 1
            continue
        if tuple(p.shape) != tuple(core_bf.shape):
            stat["skipped_shape"] += 1
            continue

        beta = _resolve_beta(beta_by_group, scope, k) if use_beta else 1.0
        # core correction: switch from beta=1 (probe) to beta_m
        core_delta = (beta - 1.0) * core_bf.to(device=p.device, dtype=torch.float32)

        # residual contribution
        Uk = residual.get(f"{k}.U")
        Vk = residual.get(f"{k}.V")
        vec = residual.get(f"{k}.vec")
        shape_t = residual.get(f"{k}.shape")

        delta = core_delta
        if Uk is not None and Vk is not None and shape_t is not None:
            orig_shape = tuple(int(x) for x in shape_t.tolist())
            lr = (Uk.to(p.device, torch.float32) @
                  Vk.to(p.device, torch.float32).t())
            lr = lr.reshape(orig_shape)
            if tuple(lr.shape) == tuple(p.shape):
                delta = delta + lr
                stat["applied_residual_lr"] += 1
            else:
                stat["skipped_shape"] += 1
            del lr
        elif vec is not None and shape_t is not None:
            orig_shape = tuple(int(x) for x in shape_t.tolist())
            v = vec.to(p.device, torch.float32).reshape(orig_shape)
            if tuple(v.shape) == tuple(p.shape):
                delta = delta + v
                stat["applied_residual_vec"] += 1
            else:
                stat["skipped_shape"] += 1
            del v

        p.data.add_(delta.to(dtype=p.dtype))
        stat["applied_core_correction"] += 1

    if verbose:
        print(f"{log_prefix}   applied_core_correction={stat['applied_core_correction']} "
              f"residual_lr={stat['applied_residual_lr']} "
              f"residual_vec={stat['applied_residual_vec']}")

    del tau_core, residual
    gc.collect()
    return stat


@torch.no_grad()
def apply_task_correction_inplace_with_track_(
    model,
    bundle_root: str,
    task_name: str,
    merge_config: dict,
    *,
    verbose: bool = True,
    log_prefix: str = "[router-overlay]",
):
    """Same as apply_task_correction_inplace_, but also returns a delta dict
    that maps weight key -> CPU fp32 tensor of the change applied. Subtracting
    these deltas reverts the model to the probe state.

    Returns:
        (stat: dict, delta_dict: dict[str, torch.Tensor])
    """
    scope = merge_config.get("scope", "per_block")
    use_beta = bool(merge_config.get("use_beta", True))
    shared_path = os.path.join(bundle_root, "shared", _CORE)
    task_dir = os.path.join(bundle_root, task_name)
    resid_path = os.path.join(task_dir, _RESID)
    if verbose:
        print(f"{log_prefix} STAGE B (with-track): {task_name}")
    tau_core = load_safetensors(shared_path)
    residual = load_safetensors(resid_path)
    with open(os.path.join(task_dir, "beta.json")) as f:
        beta_by_group = json.load(f)
    sd = dict(model.state_dict())
    stat = {
        "applied_core_correction": 0,
        "applied_residual_lr": 0,
        "applied_residual_vec": 0,
        "skipped_missing": 0,
        "skipped_shape": 0,
    }
    delta_dict = {}  # CPU fp32 deltas for revert

    for k, core_bf in tau_core.items():
        p = sd.get(k)
        if p is None:
            stat["skipped_missing"] += 1
            continue
        if tuple(p.shape) != tuple(core_bf.shape):
            stat["skipped_shape"] += 1
            continue
        beta = _resolve_beta(beta_by_group, scope, k) if use_beta else 1.0
        core_delta = (beta - 1.0) * core_bf.to(device=p.device, dtype=torch.float32)
        Uk = residual.get(f"{k}.U")
        Vk = residual.get(f"{k}.V")
        vec = residual.get(f"{k}.vec")
        shape_t = residual.get(f"{k}.shape")
        delta = core_delta
        if Uk is not None and Vk is not None and shape_t is not None:
            orig_shape = tuple(int(x) for x in shape_t.tolist())
            lr = (Uk.to(p.device, torch.float32) @
                  Vk.to(p.device, torch.float32).t())
            lr = lr.reshape(orig_shape)
            if tuple(lr.shape) == tuple(p.shape):
                delta = delta + lr
                stat["applied_residual_lr"] += 1
            else:
                stat["skipped_shape"] += 1
            del lr
        elif vec is not None and shape_t is not None:
            orig_shape = tuple(int(x) for x in shape_t.tolist())
            v = vec.to(p.device, torch.float32).reshape(orig_shape)
            if tuple(v.shape) == tuple(p.shape):
                delta = delta + v
                stat["applied_residual_vec"] += 1
            else:
                stat["skipped_shape"] += 1
            del v
        # Apply
        applied = delta.to(dtype=p.dtype)
        p.data.add_(applied)
        # Track in CPU fp32 (saves memory vs fp32 GPU; revert moves back to GPU)
        delta_dict[k] = delta.detach().to(device="cpu", dtype=torch.bfloat16).clone()
        stat["applied_core_correction"] += 1

    if verbose:
        print(f"{log_prefix}   applied={stat['applied_core_correction']} "
              f"residual_lr={stat['applied_residual_lr']} "
              f"tracked={len(delta_dict)}")
    del tau_core, residual
    gc.collect()
    return stat, delta_dict


@torch.no_grad()
def revert_delta_inplace_(model, delta_dict, *, verbose: bool = False,
                          log_prefix: str = "[router-overlay]"):
    """Subtract delta_dict from model (revert Stage B).

    Restores probe state (theta_0 + C_T).
    """
    sd = dict(model.state_dict())
    n = 0
    for k, dcpu in delta_dict.items():
        p = sd.get(k)
        if p is None:
            continue
        p.data.sub_(dcpu.to(device=p.device, dtype=p.dtype))
        n += 1
    if verbose:
        print(f"{log_prefix} reverted {n} keys")
    return n
