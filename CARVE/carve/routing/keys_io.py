"""
ConcordRouter V2 — routing key I/O.

Routing keys are reused from V1's stored residual factors.
For each weight tensor k stored as ({key}.U, {key}.V, {key}.shape) in
residual.safetensors:
    P_m^(k) = {key}.U      shape (m, r)     left factor (output subspace)
    Q_m^(k) = {key}.V      shape (n, r)     right factor (input subspace)

Routing uses Q (= V factor) only. P (= U factor) is loaded for ablation.

Convention reminder (PyTorch linear layer, weight W shape (out, in)):
    Y = X @ W.T
    correction:    delta_W = U @ V.T = P @ Q.T,  shape (out, in)
    response:      delta_Y = X @ delta_W.T = X @ Q @ P.T
    => X first hits Q (input subspace) — Q is the natural routing key.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file as load_safetensors


def _key_layer_idx(key: str) -> int:
    """Parse LLM block index from key. Returns -1 if not an LLM block weight.

    Mirrors common_merge.classify_key for self-contained use.
    """
    # OpenVLA / Prismatic format: ...language_model.model.layers.{i}.{rest}
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


def _matches_filter(key: str, weight_filter: List[str]) -> bool:
    """Check if key matches one of the weight type substrings."""
    return any(f in key for f in weight_filter)


def collect_routing_layer_keys(
    residual_path: str,
    routing_layer_indices: List[int],
    weight_filter: List[str],
) -> List[str]:
    """List residual base keys (without .U/.V suffix) that match routing config."""
    sd = load_safetensors(residual_path)
    base_keys = set()
    for k in sd.keys():
        if k.endswith(".U") or k.endswith(".V") or k.endswith(".shape") or k.endswith(".vec"):
            base = k.rsplit(".", 1)[0]
        else:
            continue
        base_keys.add(base)

    selected = []
    for base in sorted(base_keys):
        # must be 2D factorized (has .U and .V), not .vec
        if f"{base}.U" not in sd or f"{base}.V" not in sd:
            continue
        layer_idx = _key_layer_idx(base)
        if layer_idx not in routing_layer_indices:
            continue
        if not _matches_filter(base, weight_filter):
            continue
        selected.append(base)
    return selected


_STEP1_FALLBACK = "routing_keys_step1.safetensors"


def load_routing_keys(
    bundle_root: str,
    task_name: str,
    routing_layer_indices: List[int],
    weight_filter: List[str],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Load Q factors (routing keys) for one task.

    Strategy:
      1) First try residual.safetensors (V1's stored P/Q factors).
      2) Fallback: if residual is empty (typical for step_1 admissions where
         alpha_1 = 1 absorbs tau_1 entirely into the core), look for
         routing_keys_step1.safetensors produced by build_routing_keys_step1.py.

    Returns:
        Dict mapping base_key -> Q factor tensor, shape (n, r).
    """
    residual_path = os.path.join(bundle_root, task_name, "residual.safetensors")
    if not os.path.exists(residual_path):
        raise FileNotFoundError(f"missing residual.safetensors: {residual_path}")
    sd = load_safetensors(residual_path)

    selected = collect_routing_layer_keys(
        residual_path, routing_layer_indices, weight_filter
    )

    keys_out: Dict[str, torch.Tensor] = {}
    for base in selected:
        V = sd[f"{base}.V"]  # shape (n, r)
        keys_out[base] = V.to(device=device, dtype=dtype)

    # Fallback for step_1 tasks (empty residual)
    if not keys_out:
        fb_path = os.path.join(bundle_root, task_name, _STEP1_FALLBACK)
        if os.path.exists(fb_path):
            sd_fb = load_safetensors(fb_path)
            base_keys = set()
            for k in sd_fb.keys():
                if k.endswith(".U") or k.endswith(".V") or k.endswith(".shape"):
                    base_keys.add(k.rsplit(".", 1)[0])
            for base in sorted(base_keys):
                if f"{base}.V" not in sd_fb:
                    continue
                if _key_layer_idx(base) not in routing_layer_indices:
                    continue
                if not _matches_filter(base, weight_filter):
                    continue
                keys_out[base] = sd_fb[f"{base}.V"].to(device=device, dtype=dtype)
    return keys_out


def load_routing_keys_with_p(
    bundle_root: str,
    task_name: str,
    routing_layer_indices: List[int],
    weight_filter: List[str],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Load both P (U) and Q (V) factors for full-response routing ablation.

    Returns:
        Dict mapping base_key -> (P, Q),  P shape (m, r),  Q shape (n, r).

    Same step_1 fallback strategy as load_routing_keys.
    """
    residual_path = os.path.join(bundle_root, task_name, "residual.safetensors")
    sd = load_safetensors(residual_path)
    selected = collect_routing_layer_keys(
        residual_path, routing_layer_indices, weight_filter
    )
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for base in selected:
        U = sd[f"{base}.U"].to(device=device, dtype=dtype)
        V = sd[f"{base}.V"].to(device=device, dtype=dtype)
        out[base] = (U, V)

    if not out:
        fb_path = os.path.join(bundle_root, task_name, _STEP1_FALLBACK)
        if os.path.exists(fb_path):
            sd_fb = load_safetensors(fb_path)
            base_keys = set()
            for k in sd_fb.keys():
                if k.endswith(".U") or k.endswith(".V") or k.endswith(".shape"):
                    base_keys.add(k.rsplit(".", 1)[0])
            for base in sorted(base_keys):
                if f"{base}.U" not in sd_fb or f"{base}.V" not in sd_fb:
                    continue
                if _key_layer_idx(base) not in routing_layer_indices:
                    continue
                if not _matches_filter(base, weight_filter):
                    continue
                U = sd_fb[f"{base}.U"].to(device=device, dtype=dtype)
                V = sd_fb[f"{base}.V"].to(device=device, dtype=dtype)
                out[base] = (U, V)
    return out


def load_all_tasks_routing_keys(
    bundle_root: str,
    task_names: List[str],
    routing_layer_indices: List[int],
    weight_filter: List[str],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load Q factors for all tasks. Returns task -> {base_key -> Q}."""
    return {
        m: load_routing_keys(
            bundle_root, m, routing_layer_indices, weight_filter,
            device=device, dtype=dtype,
        )
        for m in task_names
    }


def list_admitted_tasks(bundle_root: str) -> List[str]:
    """Return task names from arrival_order.json under bundle_root/shared/."""
    import json
    arrival_path = os.path.join(bundle_root, "shared", "arrival_order.json")
    if os.path.exists(arrival_path):
        with open(arrival_path) as f:
            order = json.load(f)
        if isinstance(order, list):
            return [str(t) for t in order]
        if isinstance(order, dict) and "task_order" in order:
            return [str(t) for t in order["task_order"]]
    # fallback: enumerate task_dirs
    tasks = []
    for entry in sorted(os.listdir(bundle_root)):
        if entry == "shared":
            continue
        p = os.path.join(bundle_root, entry)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "residual.safetensors")):
            tasks.append(entry)
    return tasks
