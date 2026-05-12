"""
ConcordRouter V2 — recover routing keys for step_1 tasks.

Problem:
  When task t=1 is admitted with alpha_1=1.0, EMA assigns 100% to tau_1, so
  E_1 = tau_1 - C_1 = 0. The residual.safetensors file contains zero entries,
  so no routing keys can be loaded for that task.

Solution:
  Re-derive Q factors directly from the original expert checkpoint:
      tau_1 = theta_expert - theta_0
      U, S, V = svd_lowrank(tau_1[k], q=rank_max)
      P = U * sqrt(S),  Q = V * sqrt(S)
  Save Q factors to:
      {bundle_root}/{task_name}/routing_keys_step1.safetensors
  with the same key naming convention as residual.safetensors:
      {weight_key}.U   ({m}, r)
      {weight_key}.V   ({n}, r)         <- the routing key
      {weight_key}.shape  (ndim,)

This is a one-time post-hoc step; admission state is not modified.

Usage:
  python build_routing_keys_step1.py \
      --bundle /shared/.../cmm_lr_r128_t10 \
      --base_ckpt openvla/openvla-7b \
      --routing_layers 16,17,18,19,20 \
      --weight_filter self_attn.v_proj.weight,self_attn.o_proj.weight \
      --rank_max 128

The script auto-detects which admitted tasks are at step_1 (by reading
stats.json `arrival_step` or `residual_tensor_entries == 0`) and processes
only those. Tasks already having a non-empty residual are left untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MERGE_SRC_DEFAULT = os.path.dirname(_THIS_DIR)  # package root
for _p in (_THIS_DIR, _MERGE_SRC_DEFAULT, os.environ.get("MERGEVLA_SRC", "")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

def is_step1_task(bundle_root: str, task_name: str) -> bool:
    """Return True if this task was admitted at step 1 (empty residual)."""
    stats_path = os.path.join(bundle_root, task_name, "stats.json")
    if not os.path.exists(stats_path):
        return False
    with open(stats_path) as f:
        stats = json.load(f)
    diag = stats.get("diagnostics", {})
    if int(diag.get("arrival_step", -1)) == 1:
        return True
    if int(diag.get("residual_tensor_entries", 1)) == 0:
        return True
    return False


def list_admitted_tasks(bundle_root: str) -> List[str]:
    arrival_path = os.path.join(bundle_root, "shared", "arrival_order.json")
    if os.path.exists(arrival_path):
        with open(arrival_path) as f:
            order = json.load(f)
        if isinstance(order, list):
            return [str(t) for t in order]
        if isinstance(order, dict) and "task_order" in order:
            return [str(t) for t in order["task_order"]]
    out = []
    for entry in sorted(os.listdir(bundle_root)):
        if entry == "shared":
            continue
        if os.path.isdir(os.path.join(bundle_root, entry)):
            out.append(entry)
    return out


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------

def parse_layer_idx(key: str) -> int:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


def matches_weight_filter(key: str, weight_filter: List[str]) -> bool:
    return any(f in key for f in weight_filter)


def select_routing_keys(
    base_sd: Dict[str, torch.Tensor],
    layers: List[int],
    weight_filter: List[str],
) -> List[str]:
    """List weight keys (full names ending in .weight) that match routing config."""
    out = []
    for k in base_sd.keys():
        if parse_layer_idx(k) not in layers:
            continue
        if not matches_weight_filter(k, weight_filter):
            continue
        if base_sd[k].dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            continue
        out.append(k)
    return sorted(out)


# -----------------------------------------------------------------------------
# Expert / base checkpoint loading
# -----------------------------------------------------------------------------

def load_expert_state_dict(
    expert_ckpt: str,
    keys_needed: List[str],
) -> Dict[str, torch.Tensor]:
    """Load only the needed keys from a HuggingFace-style checkpoint.

    Tries safetensors index first, then falls back to a full load via
    transformers (last resort).
    """
    # 1) sharded safetensors with index
    index_path = os.path.join(expert_ckpt, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        # group keys by their shard file
        shards: Dict[str, List[str]] = {}
        for k in keys_needed:
            if k not in weight_map:
                continue
            shards.setdefault(weight_map[k], []).append(k)

        out: Dict[str, torch.Tensor] = {}
        for shard_file, ks in shards.items():
            sd = load_safetensors(os.path.join(expert_ckpt, shard_file))
            for k in ks:
                if k in sd:
                    out[k] = sd[k]
            del sd
        return out

    # 2) single safetensors
    single = os.path.join(expert_ckpt, "model.safetensors")
    if os.path.exists(single):
        sd = load_safetensors(single)
        return {k: sd[k] for k in keys_needed if k in sd}

    # 3) bin shard (rare for these models)
    bin_index = os.path.join(expert_ckpt, "pytorch_model.bin.index.json")
    if os.path.exists(bin_index):
        with open(bin_index) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shards: Dict[str, List[str]] = {}
        for k in keys_needed:
            if k in weight_map:
                shards.setdefault(weight_map[k], []).append(k)
        out: Dict[str, torch.Tensor] = {}
        for shard_file, ks in shards.items():
            sd = torch.load(os.path.join(expert_ckpt, shard_file), map_location="cpu")
            for k in ks:
                if k in sd:
                    out[k] = sd[k]
            del sd
        return out

    raise FileNotFoundError(
        f"could not locate model weights under {expert_ckpt} "
        f"(no safetensors or bin index)"
    )


def resolve_expert_ckpt_path(
    expert_id: str,
    expert_ckpt_root: str = "",
) -> str:
    """Map an expert identifier to a local path or HF id.

    expert_id may be "openvla/openvla-7b-finetuned-libero-spatial" or
    a directory under expert_ckpt_root.
    """
    if os.path.isdir(expert_id):
        return expert_id
    if expert_ckpt_root:
        candidate = os.path.join(expert_ckpt_root, expert_id)
        if os.path.isdir(candidate):
            return candidate
    # Default HF cache layout under HF_HOME
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    parts = expert_id.split("/")
    if len(parts) == 2:
        hub_dir = os.path.join(hf_home, "hub", f"models--{parts[0]}--{parts[1]}")
        snapshots = os.path.join(hub_dir, "snapshots")
        if os.path.isdir(snapshots):
            for snap in sorted(os.listdir(snapshots)):
                p = os.path.join(snapshots, snap)
                if os.path.isdir(p):
                    return p
    raise FileNotFoundError(
        f"could not resolve expert checkpoint for id={expert_id!r}; "
        f"pass --expert_ckpt_root or use an absolute directory path."
    )


# Default expert mapping for OpenVLA LIBERO bundles. Override via --expert_map.
DEFAULT_EXPERT_MAP = {
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",
    "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla/openvla-7b-finetuned-libero-10",
}


# -----------------------------------------------------------------------------
# SVD
# -----------------------------------------------------------------------------

@torch.no_grad()
def svd_routing_factors(
    delta: torch.Tensor,
    rank_max: int,
    *,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """SVD a 2D+ delta tensor; return (U_sqrtS, V_sqrtS, original_shape).

    Mirrors continual_io_lr.lowrank_residual_entries layout:
        flatten to 2D (m, n_flat), svd, take top-r, bake sqrt(S) into both
        factors. The caller saves these as {key}.U / {key}.V.
    """
    orig_shape = tuple(delta.shape)
    if delta.ndim < 2:
        raise ValueError(f"expected 2D+ tensor, got shape {orig_shape}")
    if delta.ndim == 2:
        mat = delta
    else:
        mat = delta.reshape(orig_shape[0], -1)
    mat32 = mat.to(device=device, dtype=torch.float32)
    r = min(rank_max, min(mat32.shape) - 1)
    if r < 1:
        raise ValueError(f"rank too small for shape {tuple(mat32.shape)}")
    # Randomized low-rank SVD; matches V1 admit code path.
    U, S, V = torch.svd_lowrank(mat32, q=r)
    sqrtS = S.clamp_min(0.0).sqrt()
    U_baked = U * sqrtS.unsqueeze(0)
    V_baked = V * sqrtS.unsqueeze(0)
    return U_baked.to(torch.bfloat16), V_baked.to(torch.bfloat16), orig_shape


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_routing_keys_for_task(
    bundle_root: str,
    task_name: str,
    base_ckpt_path: str,
    expert_ckpt_path: str,
    routing_layers: List[int],
    weight_filter: List[str],
    rank_max: int,
    *,
    device: str = "cpu",
    overwrite: bool = False,
    log_prefix: str = "[build-routing]",
) -> str:
    """Build and save routing_keys_step1.safetensors for a single task."""
    out_path = os.path.join(bundle_root, task_name, "routing_keys_step1.safetensors")
    if os.path.exists(out_path) and not overwrite:
        print(f"{log_prefix} {task_name}: already exists, skipping ({out_path})")
        return out_path

    print(f"{log_prefix} {task_name}: loading base from {base_ckpt_path}")
    print(f"{log_prefix} {task_name}: loading expert from {expert_ckpt_path}")

    # Decide which keys we need (probe the expert index first)
    # We look up keys via base_sd (smaller load): just those at routing layers
    # matching the filter.
    # Probe: load shapes only, then pull relevant keys.
    probe_index = os.path.join(base_ckpt_path, "model.safetensors.index.json")
    if os.path.exists(probe_index):
        with open(probe_index) as f:
            base_index = json.load(f)
        all_keys = list(base_index["weight_map"].keys())
    else:
        # full base load (cost OK; this script is rare)
        base_sd_full = load_expert_state_dict(base_ckpt_path,
                                              keys_needed=[])  # full
        if not base_sd_full:
            # fallback: load every safetensors in dir
            for fn in sorted(os.listdir(base_ckpt_path)):
                if fn.endswith(".safetensors"):
                    base_sd_full.update(
                        load_safetensors(os.path.join(base_ckpt_path, fn))
                    )
        all_keys = list(base_sd_full.keys())
        del base_sd_full

    keys_needed = [
        k for k in all_keys
        if parse_layer_idx(k) in routing_layers
        and matches_weight_filter(k, weight_filter)
    ]
    print(f"{log_prefix} {task_name}: {len(keys_needed)} target weight keys")
    if not keys_needed:
        print(f"{log_prefix} {task_name}: no matching keys, nothing to do")
        return ""

    base_sd = load_expert_state_dict(base_ckpt_path, keys_needed)
    expert_sd = load_expert_state_dict(expert_ckpt_path, keys_needed)
    print(f"{log_prefix} {task_name}: base_sd keys={len(base_sd)}, "
          f"expert_sd keys={len(expert_sd)}")

    out_tensors: Dict[str, torch.Tensor] = {}
    skipped = 0
    for k in keys_needed:
        if k not in base_sd or k not in expert_sd:
            skipped += 1
            continue
        b = base_sd[k].to(device=device, dtype=torch.float32)
        e = expert_sd[k].to(device=device, dtype=torch.float32)
        delta = e - b
        try:
            U, V, shape = svd_routing_factors(delta, rank_max=rank_max, device=device)
        except Exception as ex:
            print(f"{log_prefix} {task_name}: SVD failed on {k}: {ex}")
            skipped += 1
            continue
        out_tensors[f"{k}.U"] = U
        out_tensors[f"{k}.V"] = V
        out_tensors[f"{k}.shape"] = torch.tensor(list(shape), dtype=torch.int64)
        del b, e, delta, U, V

    # Save
    save_safetensors(out_tensors, out_path)
    print(f"{log_prefix} {task_name}: saved {len(out_tensors)} tensors to {out_path}")
    print(f"{log_prefix} {task_name}: skipped {skipped} / {len(keys_needed)}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Bundle root path")
    ap.add_argument("--base_ckpt", required=True,
                    help="Base VLA checkpoint id or path")
    ap.add_argument("--expert_ckpt_root", default="",
                    help="Optional root for resolving expert ids to local paths")
    ap.add_argument("--routing_layers", default="16,17,18,19,20",
                    help="Comma-separated LLM block indices")
    ap.add_argument("--weight_filter",
                    default="self_attn.v_proj.weight,self_attn.o_proj.weight",
                    help="Comma-separated weight key substrings")
    ap.add_argument("--rank_max", type=int, default=128)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--expert_map_json", default="",
                    help="JSON file mapping task_name -> expert ckpt id/path "
                         "(overrides DEFAULT_EXPERT_MAP)")
    ap.add_argument("--task", default="",
                    help="Process only this task (default: auto-detect step_1 tasks)")
    args = ap.parse_args()

    layers = [int(x) for x in args.routing_layers.split(",") if x.strip()]
    wfilter = [w.strip() for w in args.weight_filter.split(",") if w.strip()]

    expert_map = dict(DEFAULT_EXPERT_MAP)
    if args.expert_map_json:
        with open(args.expert_map_json) as f:
            expert_map.update(json.load(f))

    base_ckpt_path = resolve_expert_ckpt_path(args.base_ckpt, args.expert_ckpt_root)
    print(f"[build-routing] base ckpt: {base_ckpt_path}")

    admitted = list_admitted_tasks(args.bundle)
    if args.task:
        target_tasks = [args.task] if args.task in admitted else []
    else:
        target_tasks = [t for t in admitted if is_step1_task(args.bundle, t)]

    if not target_tasks:
        print("[build-routing] no step_1 tasks needing recovery")
        return

    print(f"[build-routing] target tasks: {target_tasks}")

    for task in target_tasks:
        if task not in expert_map:
            print(f"[build-routing] {task}: no expert mapping, skipping. "
                  f"Provide --expert_map_json with an entry for this task.")
            continue
        expert_id = expert_map[task]
        try:
            expert_path = resolve_expert_ckpt_path(expert_id, args.expert_ckpt_root)
        except FileNotFoundError as e:
            print(f"[build-routing] {task}: {e}")
            continue
        build_routing_keys_for_task(
            bundle_root=args.bundle,
            task_name=task,
            base_ckpt_path=base_ckpt_path,
            expert_ckpt_path=expert_path,
            routing_layers=layers,
            weight_filter=wfilter,
            rank_max=args.rank_max,
            device=args.device,
            overwrite=args.overwrite,
        )

    print("[build-routing] done")


if __name__ == "__main__":
    main()
