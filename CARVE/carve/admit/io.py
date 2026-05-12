"""
CMM LR-Residual I/O helpers.

Design:
  - Shared tau_core is identical to v2 (bf16 on disk / in memory).
  - Per-task residual is stored as a low-rank factorization per key:
        U_k in R^(m x r_k), V_k in R^(n' x r_k), both bf16
        where the key is reshaped to (m, n') = (shape[0], prod(shape[1:])).
    1D keys (biases, norm weights) have no low-rank factorization; we store the
    full vector in bf16 under {key}.vec to keep forward reconstruction trivial.
  - `rank_max` is a global ceiling. `rank_adaptive_threshold` (in [0, 1)) picks
    the effective rank per key based on normalised singular-value decay.
  - Shape is saved per key so the overlay can reshape back without peeking at
    the base state dict.

On-disk layout under <task>/:
  residual.safetensors
    {key}.U     (m, r_k)  bf16     # 2D+ keys
    {key}.V     (n', r_k) bf16     # 2D+ keys
    {key}.vec   (n,)      bf16     # 1D keys (bias/norm)
    {key}.shape (ndim,)   int64    # original shape
  beta.json
  tau_m_norms.json
  ranks.json                       # {key: effective_rank or 'vec'}
  stats.json
"""

from __future__ import annotations

import gc
import json
import math
import os
import tempfile
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from carve.admit.sparse_io import report_storage

_SHARED = "shared"
_CORE = "tau_core.safetensors"
_RESID = "residual.safetensors"


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    try:
        return float(o)
    except Exception:
        return str(o)


# ---------------------------------------------------------------------
# Shared bundle helpers (same layout as v2)
# ---------------------------------------------------------------------
def shared_dir(bundle_root: str) -> str:
    return os.path.join(bundle_root, _SHARED)


def shared_exists(bundle_root: str) -> bool:
    return os.path.exists(os.path.join(shared_dir(bundle_root), _CORE))


def atomic_save_safetensors(
    tensors: Dict[str, torch.Tensor],
    path: str,
    *,
    expected_keys: Optional[Iterable[str]] = None,
    min_bytes: int = 1024,
    sanity_name: str = "safetensors",
) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_dir = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp.", dir=final_dir)
    os.close(fd)
    try:
        save_file({k: v.contiguous() for k, v in tensors.items()}, tmp)
        size = os.path.getsize(tmp)
        if size < min_bytes:
            raise RuntimeError(f"{sanity_name}: temp file too small: {size} bytes")
        reopened = load_file(tmp)
        expected = set(expected_keys) if expected_keys is not None else set(tensors.keys())
        got = set(reopened.keys())
        if got != expected:
            missing = sorted(expected - got)[:10]
            extra = sorted(got - expected)[:10]
            raise RuntimeError(
                f"{sanity_name}: key mismatch missing={missing} extra={extra} "
                f"n_expected={len(expected)} n_got={len(got)}"
            )
        del reopened
        gc.collect()
        os.replace(tmp, path)
        return {"path": path, "bytes": size, "n_keys": len(expected)}
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(json_safe(obj), f, indent=2)
    os.replace(tmp, path)


def load_merge_config(bundle_root: str) -> dict:
    return load_json(os.path.join(shared_dir(bundle_root), "merge_config.json"), {})


def load_dataset_stats(bundle_root: str) -> dict:
    return load_json(os.path.join(shared_dir(bundle_root), "dataset_stats.json"), {}) or {}


def load_version_log(bundle_root: str) -> List[dict]:
    return load_json(os.path.join(shared_dir(bundle_root), "version_log.json"), []) or []


def load_tau_core_bf16(bundle_root: str) -> Dict[str, torch.Tensor]:
    return load_file(os.path.join(shared_dir(bundle_root), _CORE))


def save_shared_metadata(bundle_root: str, merge_config: dict, version_log: list, dataset_stats: dict) -> None:
    sd = shared_dir(bundle_root)
    save_json(os.path.join(sd, "merge_config.json"), merge_config)
    save_json(os.path.join(sd, "version_log.json"), version_log)
    save_json(os.path.join(sd, "dataset_stats.json"), dataset_stats)


def save_tau_core_atomic(
    bundle_root: str,
    tau_core_bf16: Dict[str, torch.Tensor],
    *,
    min_nonzero_ratio: float = 1e-8,
) -> dict:
    path = os.path.join(shared_dir(bundle_root), _CORE)
    stat = atomic_save_safetensors(
        {k: v.to(torch.bfloat16).contiguous() for k, v in tau_core_bf16.items()},
        path,
        expected_keys=tau_core_bf16.keys(),
        min_bytes=max(1024, len(tau_core_bf16) * 16),
        sanity_name="tau_core",
    )
    sanity = sanity_check_core(bundle_root, min_nonzero_ratio=min_nonzero_ratio)
    save_json(os.path.join(shared_dir(bundle_root), "sanity.json"), sanity)
    stat.update(sanity)
    return stat


def sanity_check_core(bundle_root: str, *, min_nonzero_ratio: float = 1e-8) -> dict:
    path = os.path.join(shared_dir(bundle_root), _CORE)
    if not os.path.exists(path):
        raise RuntimeError(f"missing tau_core: {path}")
    tensors = load_file(path)
    n_keys = len(tensors)
    n_elem = 0
    n_nonzero = 0
    abs_sum_sample = 0.0
    finite_ok = True
    for k, v in tensors.items():
        n_elem += v.numel()
        nz = (v != 0).sum().item()
        n_nonzero += int(nz)
        if abs_sum_sample < 1e30:
            vf = v.float()
            finite_ok = finite_ok and bool(torch.isfinite(vf).all().item())
            abs_sum_sample += float(vf.abs().sum().item())
        del v
    ratio = n_nonzero / max(n_elem, 1)
    out = {
        "tau_core_path": path,
        "bytes": os.path.getsize(path),
        "n_keys": n_keys,
        "n_elem": n_elem,
        "n_nonzero": n_nonzero,
        "nonzero_ratio": ratio,
        "abs_sum": abs_sum_sample,
        "finite_ok": finite_ok,
    }
    if not finite_ok:
        raise RuntimeError("tau_core sanity failed: non-finite values")
    if n_elem > 0 and ratio < min_nonzero_ratio:
        raise RuntimeError(f"tau_core sanity failed: nonzero_ratio={ratio:.3e}")
    return out


# ---------------------------------------------------------------------
# Low-rank residual: SVD & key-wise serialisation
# ---------------------------------------------------------------------
@torch.no_grad()
def lowrank_factor_key(
    residual_tensor: torch.Tensor,
    *,
    rank_max: int,
    rank_adaptive_threshold: float = 0.0,
    oversample: int = 8,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, float]:
    """Factorise a residual into U (m,r) and V (n,r) with bf16 storage.

    Returns (U_bf16, V_bf16, r_effective, reconstruction_error_rel).

    `rank_adaptive_threshold`:
        If > 0, we keep singular values s_i where s_i / s_0 > threshold,
        capped at rank_max. If 0, we use rank_max (or min(m, n)).

    Assumes residual_tensor is already 2D (m, n'). Caller must reshape.
    The factors U and V absorb sqrt(singular values) so that U @ V.T ≈ residual.
    """
    assert residual_tensor.ndim == 2, "residual_tensor must be 2D"
    m, n = residual_tensor.shape
    # Work in fp32 for stable SVD.
    mat = residual_tensor.float()
    base_norm = float(torch.linalg.norm(mat).item())
    if base_norm <= 0.0:
        return None, None, 0, 0.0

    q = min(rank_max + oversample, m, n)
    if q < 1:
        return None, None, 0, 1.0

    # randomized lowrank svd; mat may be on CPU
    try:
        U, s, V = torch.svd_lowrank(mat, q=q, niter=4)
    except Exception as e:
        # Fallback: full SVD (will be costly but correct)
        U_full, s, V_full = torch.linalg.svd(mat, full_matrices=False)
        U = U_full[:, :q]
        s = s[:q]
        V = V_full[:q, :].T  # match svd_lowrank convention (V: n x q)

    if rank_adaptive_threshold > 0 and s.numel() > 0:
        s0 = float(s[0].item())
        if s0 <= 0.0:
            return None, None, 0, 1.0
        keep = (s / s0) > rank_adaptive_threshold
        r_eff = int(keep.sum().item())
        r_eff = max(1, min(r_eff, rank_max, q))
    else:
        r_eff = min(rank_max, q)

    s_eff = s[:r_eff]
    U_eff = U[:, :r_eff]
    V_eff = V[:, :r_eff]
    # Absorb sqrt(s) into both factors.
    sqrt_s = s_eff.clamp(min=0).sqrt()
    U_scaled = (U_eff * sqrt_s).to(torch.bfloat16).contiguous()
    V_scaled = (V_eff * sqrt_s).to(torch.bfloat16).contiguous()

    # reconstruction error (relative Frobenius)
    approx = (U_scaled.float() @ V_scaled.float().T)
    err = float(torch.linalg.norm(mat - approx).item()) / max(base_norm, 1e-12)
    return U_scaled, V_scaled, int(r_eff), err


@torch.no_grad()
def lowrank_residual_entries(
    key: str,
    residual: torch.Tensor,
    *,
    rank_max: int,
    rank_adaptive_threshold: float,
    min_factorise_numel: int,
    oversample: int = 8,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Produce safetensors entries for one key's residual.

    1D keys and tiny keys are stored verbatim (bf16) under {key}.vec.
    Other keys are reshaped to 2D, SVD-factorised, stored as {key}.U / {key}.V.
    Shape is saved under {key}.shape for exact reconstruction.
    """
    shape = list(residual.shape)
    shape_tensor = torch.tensor(shape, dtype=torch.int64)

    # Trivial skip: residual is exactly zero.
    if not bool((residual != 0).any().item()):
        return {}, {"rank": 0, "mode": "empty", "reconstruction_rel_err": 0.0,
                    "numel": int(residual.numel())}

    numel = int(residual.numel())

    # Small / 1D keys: store as full bf16 vector.
    if residual.ndim < 2 or numel < max(1, min_factorise_numel):
        vec = residual.flatten().to(torch.bfloat16).contiguous()
        return (
            {f"{key}.vec": vec, f"{key}.shape": shape_tensor},
            {"rank": 0, "mode": "vec", "reconstruction_rel_err": 0.0, "numel": numel},
        )

    # 2D+: reshape to (shape[0], prod(shape[1:])) and factorise.
    mat = residual.reshape(shape[0], -1).contiguous()
    U, V, r_eff, err = lowrank_factor_key(
        mat,
        rank_max=rank_max,
        rank_adaptive_threshold=rank_adaptive_threshold,
        oversample=oversample,
    )
    if U is None or V is None or r_eff <= 0:
        return {}, {"rank": 0, "mode": "empty", "reconstruction_rel_err": 0.0,
                    "numel": numel}
    return (
        {
            f"{key}.U": U,
            f"{key}.V": V,
            f"{key}.shape": shape_tensor,
        },
        {"rank": r_eff, "mode": "lr", "reconstruction_rel_err": err, "numel": numel,
         "m": int(mat.shape[0]), "n": int(mat.shape[1])},
    )


def iter_residual_keys(flat: Dict[str, torch.Tensor]):
    """Iterate keys that have a full entry (either .U/.V or .vec) with shape."""
    seen = set()
    for ik in flat.keys():
        base = None
        if ik.endswith(".U"):
            base = ik[:-2]
        elif ik.endswith(".vec"):
            base = ik[:-4]
        if base is None or base in seen:
            continue
        seen.add(base)
        shape_key = f"{base}.shape"
        if shape_key not in flat:
            continue
        yield base


def save_task_lr(
    task_dir: str,
    *,
    residual_flat: Dict[str, torch.Tensor],
    beta_by_group: dict,
    tau_m_norms_by_group: dict,
    ranks: dict,
    stats: Optional[dict] = None,
) -> None:
    os.makedirs(task_dir, exist_ok=True)
    atomic_save_safetensors(
        residual_flat,
        os.path.join(task_dir, _RESID),
        expected_keys=residual_flat.keys(),
        min_bytes=0,
        sanity_name="lr_residual",
    )
    save_json(os.path.join(task_dir, "beta.json"), beta_by_group)
    save_json(os.path.join(task_dir, "tau_m_norms.json"), tau_m_norms_by_group)
    save_json(os.path.join(task_dir, "ranks.json"), ranks)
    if stats is not None:
        save_json(os.path.join(task_dir, "stats.json"), stats)


def update_task_beta(task_dir: str, beta_by_group: dict) -> None:
    save_json(os.path.join(task_dir, "beta.json"), beta_by_group)


def report_cmm_bundle(bundle_root: str, task_names: List[str]) -> dict:
    return report_storage(bundle_root, task_names)
