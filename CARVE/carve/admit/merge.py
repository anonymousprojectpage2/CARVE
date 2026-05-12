"""
CMM LR-Residual: streaming protected core + per-task low-rank residual.

Algorithm for admitting task t:
  tau_t = theta_t - theta_0
  protected = (tau_core == 0) OR (sign(tau_core) == sign(tau_t))
  core_new[protected] = (1 - alpha_t) * core_old[protected] + alpha_t * tau_t[protected]
  core_new[~protected] = core_old[~protected]
  alpha_t = 1 / sqrt(arrival_step)

  residual_t = tau_t - core_new           # "what the shared core could not absorb"
  for each key k:
      if residual_t[k] is 1D or small:    save as bf16 vector
      else:                               SVD-factorise into U_k (m,r), V_k (n',r)

Eval overlay:
  theta_m = theta_0 + beta_m * tau_core + residual_m
           where residual_m[k] = U_m[k] @ V_m[k].T (reshaped) or full vector.

Memory discipline:
  - tau_core is kept bf16 in memory and on disk.
  - For each key we temporarily allocate fp32 copies of base, task, core, tau_t,
    residual; we free them per-key before moving on.
  - Randomized SVD via torch.svd_lowrank — cheap even for 4096x11008.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import torch

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SELF_DIR, os.environ.get("MERGEVLA_SRC", _SELF_DIR)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from carve.admit.utils import (  # noqa: E402
    register_openvla,
    load_base_sd,
    load_model,
    free_memory,
    classify_key,
    aggregate_dataset_statistics,
    FLOAT_DTYPES,
)
from carve.admit.io import (  # noqa: E402
    shared_dir,
    shared_exists,
    load_tau_core_bf16,
    load_merge_config,
    load_version_log,
    load_dataset_stats,
    save_shared_metadata,
    save_tau_core_atomic,
    save_task_lr,
    update_task_beta,
    lowrank_residual_entries,
    report_cmm_bundle,
    save_json,
)


METHOD_TAG = "CMM-LR"


def block_group(key: str) -> str:
    info = classify_key(key)
    if info["layer_idx"] >= 0:
        return f"llm_block_{info['layer_idx']:02d}"
    return f"non_llm_{info['module']}"


def group_for_scope(key: str, scope: str) -> str:
    if scope == "global":
        return "global"
    if scope == "per_key":
        return key
    return block_group(key)


def alpha_schedule(t: int, mode: str) -> float:
    if mode == "constant":
        return 1.0
    if mode == "inv_t":
        return 1.0 / max(t, 1)
    if mode == "inv_sqrt":
        return 1.0 / math.sqrt(max(t, 1))
    raise ValueError(f"unknown alpha_mode={mode}")


def derive_beta(
    tau_norms: Dict[str, float],
    core_norms: Dict[str, float],
    gamma: float,
    scope: str,
) -> Dict[str, float]:
    """Scale core so that its mass per group matches tau's mass per group."""
    if scope == "global":
        t = float(tau_norms.get("global", 0.0))
        d = float(core_norms.get("global", 0.0))
        return {"global": min(1.0, gamma * ((t / d) if d > 1e-12 else 1.0))}

    out = {}
    for g, t in tau_norms.items():
        if g == "global":
            continue
        d = float(core_norms.get(g, 0.0))
        out[g] = min(1.0, gamma * ((float(t) / d) if d > 1e-12 else 1.0))
    return out


def init_empty_core_from_base(base_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in base_sd.items():
        if v.dtype in FLOAT_DTYPES:
            out[k] = torch.zeros_like(v, dtype=torch.bfloat16, device="cpu").contiguous()
    return out


def ensure_bundle(
    bundle_root: str,
    base_sd: Dict[str, torch.Tensor],
    *,
    base_ckpt: str,
    task_ckpt: str,
    gamma: float,
    alpha_mode: str,
    scope: str,
    rank_max: int,
    rank_adaptive_threshold: float,
    min_factorise_numel: int,
    use_beta: bool,
) -> None:
    if shared_exists(bundle_root):
        return
    os.makedirs(shared_dir(bundle_root), exist_ok=True)
    cfg = {
        "method": METHOD_TAG,
        "version": "cmm_lr_v1",
        "base_checkpoint": base_ckpt,
        "gamma": gamma,
        "alpha_mode": alpha_mode,
        "scope": scope,
        "task_order": [],
        "core_dtype": "bfloat16",
        "residual_dtype": "bfloat16",
        "rank_max": rank_max,
        "rank_adaptive_threshold": rank_adaptive_threshold,
        "min_factorise_numel": min_factorise_numel,
        "use_beta": bool(use_beta),
        "mask_rule": "protected_same_sign_or_empty",
        "update_rule": "ema_on_protected_coordinates",
        "residual_rule": "keywise_low_rank_svd",
    }
    try:
        dataset_stats = aggregate_dataset_statistics([task_ckpt])
    except Exception as e:
        print(f"[init] WARN: dataset statistics unavailable: {type(e).__name__}: {e}")
        dataset_stats = {}
    save_shared_metadata(bundle_root, cfg, [], dataset_stats)
    tau_core = init_empty_core_from_base(base_sd)
    print(f"[init] writing empty bf16 tau_core with {len(tau_core)} keys")
    save_tau_core_atomic(bundle_root, tau_core, min_nonzero_ratio=0.0)
    del tau_core
    gc.collect()
    print(f"[init] created CMM LR bundle: {bundle_root}")


@torch.no_grad()
def recompute_beta_for_task(
    task_dir: str,
    tau_core_bf16: Dict[str, torch.Tensor],
    base_sd: Dict[str, torch.Tensor],
    *,
    gamma: float,
    scope: str,
) -> Dict[str, float]:
    """Update beta_m after tau_core has changed.

    LR path: we do not store a mask, so the 'core norm' is the full core norm per group.
    """
    norms_path = os.path.join(task_dir, "tau_m_norms.json")
    if not os.path.exists(norms_path):
        raise FileNotFoundError(f"missing tau_m_norms.json: {task_dir}")
    with open(norms_path) as f:
        tau_norms = json.load(f)

    sq = defaultdict(float)
    sq_global = 0.0
    for k, core_bf in tau_core_bf16.items():
        if k not in base_sd:
            continue
        core_f = core_bf.float()
        s = float(core_f.pow(2).sum().item())
        if scope == "per_key":
            sq[k] += s
        elif scope == "per_block":
            sq[block_group(k)] += s
        sq_global += s
        del core_f
    core_norms = {g: math.sqrt(v) for g, v in sq.items()}
    core_norms["global"] = math.sqrt(sq_global)
    beta = derive_beta(tau_norms, core_norms, gamma, scope)
    update_task_beta(task_dir, beta)
    return beta


@torch.no_grad()
def admit_expert(
    *,
    bundle_root: str,
    task_name: str,
    task_ckpt: str,
    base_ckpt: str = "openvla/openvla-7b",
    gamma: float = 1.0,
    alpha_mode: str = "inv_sqrt",
    scope: str = "per_block",
    rank_max: int = 64,
    rank_adaptive_threshold: float = 0.0,
    min_factorise_numel: int = 4096,
    use_beta: bool = True,
    force: bool = False,
) -> dict:
    register_openvla()

    print(f"[admit] loading base state dict: {base_ckpt}")
    base_sd = load_base_sd(base_ckpt)

    ensure_bundle(
        bundle_root,
        base_sd,
        base_ckpt=base_ckpt,
        task_ckpt=task_ckpt,
        gamma=gamma,
        alpha_mode=alpha_mode,
        scope=scope,
        rank_max=rank_max,
        rank_adaptive_threshold=rank_adaptive_threshold,
        min_factorise_numel=min_factorise_numel,
        use_beta=use_beta,
    )

    merge_config = load_merge_config(bundle_root)
    version_log = load_version_log(bundle_root)
    dataset_stats = load_dataset_stats(bundle_root)
    task_order = list(merge_config.get("task_order", []))
    if task_name in task_order and not force:
        raise RuntimeError(f"task {task_name!r} already admitted; pass --force to overwrite task dir")
    if force and task_name in task_order:
        task_order = [t for t in task_order if t != task_name]
        version_log = [v for v in version_log if v.get("task") != task_name]

    for name, value in [("base_checkpoint", base_ckpt), ("scope", scope)]:
        old = merge_config.get(name)
        if old is not None and old != value:
            raise RuntimeError(f"bundle {name}={old!r}, requested {value!r}")
    merge_config.update({
        "gamma": gamma,
        "alpha_mode": alpha_mode,
        "rank_max": rank_max,
        "rank_adaptive_threshold": rank_adaptive_threshold,
        "min_factorise_numel": min_factorise_numel,
        "use_beta": bool(use_beta),
    })

    print(f"[admit] loading tau_core bf16")
    tau_core = load_tau_core_bf16(bundle_root)
    mergeable_keys = [k for k in base_sd if base_sd[k].dtype in FLOAT_DTYPES and k in tau_core]

    t = len(task_order) + 1
    alpha_t = alpha_schedule(t, alpha_mode)
    print(
        f"[admit] task={task_name} arrival={t} alpha={alpha_t:.4f} "
        f"gamma={gamma} scope={scope} rank_max={rank_max} "
        f"adapt={rank_adaptive_threshold} min_numel={min_factorise_numel}"
    )

    print(f"[admit] loading task model: {task_ckpt}")
    model = load_model(task_ckpt)
    sd = model.state_dict()

    residual_flat: Dict[str, torch.Tensor] = {}
    ranks: Dict[str, dict] = {}

    tau_sq = defaultdict(float)
    tau_sq_global = 0.0
    core_sq = defaultdict(float)
    core_sq_global = 0.0

    n_total = 0
    n_protected = 0
    n_empty = 0
    n_same_sign = 0
    total_residual_params = 0    # sum of saved (U,V) param counts + vec counts
    sum_rel_err_weighted = 0.0   # weighted by ||tau_t||_F^2
    sum_tau_sq_for_err = 0.0

    for i, k in enumerate(mergeable_keys):
        if k not in sd or tuple(sd[k].shape) != tuple(base_sd[k].shape):
            continue
        base_k = base_sd[k]
        task_k = sd[k].detach().cpu()
        tau_t = task_k.float() - base_k.float()
        core_old = tau_core[k].float()

        core_empty = core_old.abs() <= 0.0
        same_sign = (tau_t * core_old) > 0
        protected = core_empty | same_sign

        core_new = core_old.clone()
        core_new[protected] = (1.0 - alpha_t) * core_old[protected] + alpha_t * tau_t[protected]
        tau_core[k] = core_new.to(torch.bfloat16).contiguous()

        residual_k = tau_t - core_new

        entries, rk_stats = lowrank_residual_entries(
            k,
            residual_k,
            rank_max=rank_max,
            rank_adaptive_threshold=rank_adaptive_threshold,
            min_factorise_numel=min_factorise_numel,
        )
        if entries:
            residual_flat.update(entries)
            # size accounting
            for tname, tens in entries.items():
                if tname.endswith(".shape"):
                    continue
                total_residual_params += int(tens.numel())
        ranks[k] = rk_stats

        # Norm accounting for beta (tau norm AND core norm).
        s_tau = float(tau_t.pow(2).sum().item())
        s_core = float(core_new.float().pow(2).sum().item())
        g = group_for_scope(k, scope)
        if scope != "global":
            tau_sq[g] += s_tau
            core_sq[g] += s_core
        tau_sq_global += s_tau
        core_sq_global += s_core

        # Reconstruction error accumulation (weighted by ||tau_t||^2)
        rel = float(rk_stats.get("reconstruction_rel_err", 0.0))
        sum_rel_err_weighted += rel * s_tau
        sum_tau_sq_for_err += s_tau

        n = tau_t.numel()
        n_total += n
        n_protected += int(protected.sum().item())
        n_empty += int(core_empty.sum().item())
        n_same_sign += int(same_sign.sum().item())

        if (i + 1) % 250 == 0:
            print(f"  [admit] processed {i + 1}/{len(mergeable_keys)} keys")

        del base_k, task_k, tau_t, core_old, core_new, residual_k, core_empty, same_sign, protected

    del model, sd
    free_memory()

    tau_norms = {g: math.sqrt(v) for g, v in tau_sq.items()}
    tau_norms["global"] = math.sqrt(tau_sq_global)
    core_norms = {g: math.sqrt(v) for g, v in core_sq.items()}
    core_norms["global"] = math.sqrt(core_sq_global)

    if use_beta:
        beta_t = derive_beta(tau_norms, core_norms, gamma, scope)
    else:
        # Fix beta = 1.0 per group: core is used verbatim.
        beta_t = {g: 1.0 for g in tau_norms.keys()}

    avg_rel_err = sum_rel_err_weighted / max(sum_tau_sq_for_err, 1e-12)

    diag = {
        "arrival_step": t,
        "alpha_t": alpha_t,
        "total_positions": n_total,
        "protected_positions": n_protected,
        "protected_empty": n_empty,
        "protected_same_sign": n_same_sign,
        "protected_ratio_pct": 100.0 * n_protected / max(n_total, 1),
        "residual_tensor_entries": len(residual_flat),
        "residual_total_params": total_residual_params,
        "residual_bytes_estimate": total_residual_params * 2,  # bf16
        "weighted_reconstruction_rel_err": avg_rel_err,
        "beta_global": beta_t.get("global"),
    }
    print(
        f"[diag] protected={diag['protected_ratio_pct']:.2f}% "
        f"residual_params={total_residual_params:,} "
        f"residual_bytes~{total_residual_params * 2 / 1024 ** 2:.1f}MB "
        f"recon_rel_err={avg_rel_err:.4e}"
    )

    task_dir = os.path.join(bundle_root, task_name)
    print(f"[save] task state -> {task_dir}")
    save_task_lr(
        task_dir,
        residual_flat=residual_flat,
        beta_by_group=beta_t,
        tau_m_norms_by_group=tau_norms,
        ranks=ranks,
        stats={"diagnostics": diag, "core_norms": core_norms},
    )
    del residual_flat
    gc.collect()

    # Update prior betas. For LR we just rescale betas against the new core norm.
    print("[recompute] prior beta after core update")
    for prev in task_order:
        prev_dir = os.path.join(bundle_root, prev)
        if not os.path.isdir(prev_dir):
            print(f"  [skip] missing prior dir: {prev}")
            continue
        if use_beta:
            new_beta = recompute_beta_for_task(prev_dir, tau_core, base_sd, gamma=gamma, scope=scope)
            print(f"  [✓] {prev}: beta_global={new_beta.get('global', 0.0):.4f}")
        else:
            # still write a constant beta for consistency
            beta_const = {g: 1.0 for g in tau_norms.keys()}
            update_task_beta(prev_dir, beta_const)
            print(f"  [·] {prev}: use_beta=False, beta=1.0")

    try:
        ds_new = aggregate_dataset_statistics([task_ckpt])
        if ds_new:
            dataset_stats.update(ds_new)
    except Exception as e:
        print(f"[ds_stats] WARN: skip dataset stats update: {type(e).__name__}: {e}")

    task_order.append(task_name)
    merge_config["task_order"] = task_order
    version_log.append({
        "step": t,
        "task": task_name,
        "task_ckpt": task_ckpt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "alpha_t": alpha_t,
        "rank_max": rank_max,
        "rank_adaptive_threshold": rank_adaptive_threshold,
        "use_beta": bool(use_beta),
        "diagnostics": diag,
    })
    save_shared_metadata(bundle_root, merge_config, version_log, dataset_stats)

    print("[save] tau_core atomic write + sanity")
    sanity = save_tau_core_atomic(bundle_root, tau_core, min_nonzero_ratio=1e-10)
    print(
        f"[sanity] core keys={sanity['n_keys']} nonzero_ratio={sanity['nonzero_ratio']:.6e} "
        f"bytes={sanity['bytes'] / (1024 ** 3):.2f}GB"
    )
    del tau_core, base_sd
    gc.collect()

    rpt = report_cmm_bundle(bundle_root, task_order)
    save_json(os.path.join(bundle_root, "_summary.json"),
              {"last_admit": diag, "storage": rpt, "config": merge_config})
    print(f"[storage] shared={rpt['shared_gb']:.2f}GB total={rpt['total_gb']:.2f}GB")
    for name, gb in rpt["per_task_gb"].items():
        print(f"          {name}: {gb:.3f}GB")

    return {"diagnostics": diag, "sanity": sanity, "storage": rpt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_root", required=True)
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--task_ckpt", required=True)
    ap.add_argument("--base_ckpt", default="openvla/openvla-7b")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--alpha_mode", choices=["constant", "inv_t", "inv_sqrt"], default="inv_sqrt")
    ap.add_argument("--scope", choices=["global", "per_block", "per_key"], default="per_block")
    ap.add_argument("--rank_max", type=int, default=64, help="Max singular values kept per key.")
    ap.add_argument("--rank_adaptive_threshold", type=float, default=0.0,
                    help="If > 0, keep s_i where s_i/s_0 > threshold, capped at rank_max. 0 disables.")
    ap.add_argument("--min_factorise_numel", type=int, default=4096,
                    help="Keys with numel < this (or ndim < 2) are stored as a bf16 vector instead of SVD.")
    ap.add_argument("--use_beta", type=int, default=1, help="1: compute beta (v2-like). 0: force beta=1.0.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("  CMM LR-Residual admit expert")
    print(f"  bundle={args.bundle_root}")
    print(f"  task={args.task_name}")
    print(f"  rank_max={args.rank_max} adapt={args.rank_adaptive_threshold} "
          f"min_numel={args.min_factorise_numel} use_beta={bool(args.use_beta)}")
    print(f"  gamma={args.gamma} alpha={args.alpha_mode} scope={args.scope}")
    print("=" * 72)

    admit_expert(
        bundle_root=args.bundle_root,
        task_name=args.task_name,
        task_ckpt=args.task_ckpt,
        base_ckpt=args.base_ckpt,
        gamma=args.gamma,
        alpha_mode=args.alpha_mode,
        scope=args.scope,
        rank_max=args.rank_max,
        rank_adaptive_threshold=args.rank_adaptive_threshold,
        min_factorise_numel=args.min_factorise_numel,
        use_beta=bool(args.use_beta),
        force=args.force,
    )


if __name__ == "__main__":
    main()
