"""
Shared building blocks for the three VLA merging methods (A / B / D).

Responsibilities:
  - OpenVLA HF registration and model loading
  - Compute τ_merge = Σ_m (Θ_m - Θ_0)  with identical-key caching
  - Compute per-task TA+S masks S_m = 1[|τ_m| > λ · |τ_merge - τ_m|]
  - Parameter-group classification  (trunk / head / lm_head etc.) used by Method B
  - Dataset-statistics aggregation for saving
  - Lightweight save helpers (torch.save with reporting)

Usage:
    from common_merge import (
        register_openvla, load_model, load_base_sd,
        compute_tau_merge_and_identical, compute_task_masks,
        classify_key,
        save_dict_with_report, aggregate_dataset_statistics,
        TASK_CHECKPOINTS, FLOAT_DTYPES,
    )
"""

from __future__ import annotations

import gc
import json
import os
import re
from typing import Dict, List, Set, Tuple

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
TASK_CHECKPOINTS: List[Tuple[str, str]] = [
    ("libero_spatial", "openvla/openvla-7b-finetuned-libero-spatial"),
    ("libero_object",  "openvla/openvla-7b-finetuned-libero-object"),
    ("libero_goal",    "openvla/openvla-7b-finetuned-libero-goal"),
    ("libero_10",      "openvla/openvla-7b-finetuned-libero-10"),
]

FLOAT_DTYPES = {torch.float32, torch.float16, torch.bfloat16}

DEFAULT_IDENTICAL_CACHE = "./merged_openvla/_cache/identical_keys_openvla_libero.pt"


# ---------------------------------------------------------------------
# OpenVLA registration / loading
# ---------------------------------------------------------------------
def register_openvla() -> None:
    """Register OpenVLA classes in HF Auto* registries (idempotent)."""
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ValueError:
        pass


def load_model(checkpoint_path: str):
    """Load an OpenVLA checkpoint (CPU, bf16)."""
    print(f"    [*] Loading: {checkpoint_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_base_sd(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint and return its state_dict (on CPU)."""
    model = load_model(checkpoint_path)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    free_memory()
    return sd


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Key classification — used for Method B (trunk vs head)
# ---------------------------------------------------------------------
def classify_key(key: str, num_layers: int = 32) -> dict:
    """
    Return a dict with fields:
      module : "vision", "projector", "embed", "llm_attn", "llm_mlp",
               "llm_norm", "llm_other", "lm_head", "other"
      layer_idx : int, 0-based LLM transformer block index.
                  -1 for non-LLM-layer params.

    OpenVLA LLaMA-7B has 32 transformer blocks → num_layers = 32.
    """
    m = re.search(r"(?:language_model|llm|model)\.layers\.(\d+)\.(.+)", key)
    if m:
        layer_idx = int(m.group(1))
        sub = m.group(2)
        if any(tag in sub for tag in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"]):
            module = "llm_attn"
        elif any(tag in sub for tag in ["gate_proj", "up_proj", "down_proj", "mlp"]):
            module = "llm_mlp"
        elif "norm" in sub or "layernorm" in sub:
            module = "llm_norm"
        else:
            module = "llm_other"
        return {"module": module, "layer_idx": layer_idx}

    if "lm_head" in key:
        return {"module": "lm_head", "layer_idx": -1}
    if "embed_tokens" in key or "token_embedding" in key:
        return {"module": "embed", "layer_idx": -1}
    if "vision_backbone" in key or re.search(r"vision.*layers?\.(\d+)", key):
        return {"module": "vision", "layer_idx": -1}
    if any(tag in key for tag in ["projector", "multi_modal_projector", "mm_projector"]):
        return {"module": "projector", "layer_idx": -1}

    return {"module": "other", "layer_idx": -1}


def is_head_key(key: str, head_last_n_blocks: int = 1, include_lm_head: bool = True,
                num_layers: int = 32) -> bool:
    """
    True if `key` belongs to the 'expert head' in Method B:
      - lm_head, AND/OR
      - the last `head_last_n_blocks` LLM transformer blocks.
    """
    info = classify_key(key, num_layers=num_layers)
    if include_lm_head and info["module"] == "lm_head":
        return True
    if info["layer_idx"] >= 0 and info["layer_idx"] >= num_layers - head_last_n_blocks:
        return True
    return False


# ---------------------------------------------------------------------
# Stage 1: τ_merge + identical_keys
# ---------------------------------------------------------------------
def compute_tau_merge_and_identical(
    base_sd: Dict[str, torch.Tensor],
    task_checkpoints: List[str],
    alpha: float = 1.0,
    identical_cache_path: str = DEFAULT_IDENTICAL_CACHE,
    force_recompute_identical: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
    """
    Single pass that accumulates τ_merge = α · Σ_m (Θ_m - Θ_0)  (fp32, CPU)
    while also identifying `identical_keys` where all fine-tunes match base bit-wise.

    Returns:
      tau_merge : dict[key -> fp32 CPU tensor] over non-identical keys only
      identical_keys : set of keys that are bit-equal to Θ_0 in every task
    """
    cached: Set[str] = None
    if identical_cache_path and os.path.exists(identical_cache_path) and not force_recompute_identical:
        try:
            cached = set(torch.load(identical_cache_path))
            print(f"[*] Loaded cached identical_keys ({len(cached):,} keys)")
        except Exception as e:
            print(f"[!] Cache load failed ({e}); recomputing")
            cached = None

    float_keys = [k for k in base_sd if base_sd[k].dtype in FLOAT_DTYPES]
    mergeable_keys = [k for k in float_keys if k not in cached] if cached is not None else float_keys
    print(f"[*] Tracking τ_merge over {len(mergeable_keys):,} float keys "
          f"(identical cache hit: {cached is not None})")

    tau_merge: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(base_sd[k], dtype=torch.float32) for k in mergeable_keys
    }
    differ_set: Set[str] = set() if cached is None else None

    print(f"[*] Accumulating τ_merge (α={alpha}) over {len(task_checkpoints)} ckpts")
    for i, ckpt in enumerate(task_checkpoints):
        model = load_model(ckpt)
        sd = model.state_dict()
        for k in mergeable_keys:
            if k not in sd or base_sd[k].shape != sd[k].shape:
                continue
            diff = sd[k].detach().float().cpu() - base_sd[k].float()
            tau_merge[k].add_(diff)
            if differ_set is not None and k not in differ_set:
                if bool((sd[k] != base_sd[k]).any().item()):
                    differ_set.add(k)
        del model, sd, diff
        free_memory()
        print(f"    [✓] τ_merge += τ_{i + 1}  ({i + 1}/{len(task_checkpoints)})")

    if alpha != 1.0:
        for k in tau_merge:
            tau_merge[k].mul_(alpha)

    if cached is not None:
        identical_keys = cached
    else:
        identical_keys = set(mergeable_keys) - differ_set
        for k in identical_keys:
            if k in tau_merge:
                del tau_merge[k]
        free_memory()
        if identical_cache_path:
            os.makedirs(os.path.dirname(identical_cache_path) or ".", exist_ok=True)
            torch.save(sorted(identical_keys), identical_cache_path)
            print(f"[*] Cached identical_keys → {identical_cache_path}")

    print(f"\n[stats] non-identical: {len(tau_merge):,} keys, "
          f"identical: {len(identical_keys):,} keys")
    return tau_merge, identical_keys


# ---------------------------------------------------------------------
# Stage 2: per-task TA+S mask  S_m = 1[|τ_m| > λ · |τ_merge − τ_m|]
#          (Also returns per-task τ_m so downstream methods can reuse.)
# ---------------------------------------------------------------------
def compute_task_masks_and_taus(
    base_sd: Dict[str, torch.Tensor],
    task_checkpoints_with_names: List[Tuple[str, str]],
    tau_merge: Dict[str, torch.Tensor],
    lambda_: float,
    keep_taus: bool = False,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Recompute per-task mask S_m and optionally cache τ_m for downstream methods.

    keep_taus=False  : returns (masks, {})          (memory-efficient mode)
    keep_taus=True   : returns (masks, taus)        (required by Method A/D)
    """
    print(f"\n[*] Computing per-task masks S_m  (λ={lambda_})  "
          f"keep_taus={keep_taus}")
    masks: Dict[str, Dict[str, torch.Tensor]] = {}
    taus:  Dict[str, Dict[str, torch.Tensor]] = {}

    for task_name, ckpt in task_checkpoints_with_names:
        model = load_model(ckpt)
        sd = model.state_dict()

        mask_m: Dict[str, torch.Tensor] = {}
        tau_m_store: Dict[str, torch.Tensor] = {}
        n_active, n_total = 0, 0
        for k in tau_merge:
            if k not in sd or base_sd[k].shape != sd[k].shape:
                continue
            tau_m = sd[k].detach().float().cpu() - base_sd[k].float()
            residual = tau_merge[k] - tau_m
            mask = tau_m.abs() > (lambda_ * residual.abs())
            mask_m[k] = mask

            if keep_taus:
                # Store in fp32 for precision during later norm/scaling ops.
                tau_m_store[k] = tau_m.contiguous()

            n_total += mask.numel()
            n_active += int(mask.sum().item())
            del residual, mask
            if not keep_taus:
                del tau_m

        print(f"    [✓] {task_name}: mask active ratio "
              f"{n_active / max(n_total, 1) * 100:.2f}%")
        masks[task_name] = mask_m
        if keep_taus:
            taus[task_name] = tau_m_store
        del model, sd
        free_memory()

    return masks, taus


# ---------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------
def download_dataset_statistics(checkpoint_path: str) -> dict:
    local_stats_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(local_stats_path):
        with open(local_stats_path, "r") as f:
            return json.load(f)
    local_path = hf_hub_download(repo_id=checkpoint_path, filename="dataset_statistics.json")
    with open(local_path, "r") as f:
        return json.load(f)


def aggregate_dataset_statistics(task_checkpoints: List[str]) -> dict:
    merged: dict = {}
    for ckpt in task_checkpoints:
        try:
            merged.update(download_dataset_statistics(ckpt))
        except Exception as e:
            print(f"    [!] dataset_statistics load failed for {ckpt}: {e}")
    return merged


def save_dataset_statistics(save_root: str, task_checkpoints: List[str]) -> str:
    stats = aggregate_dataset_statistics(task_checkpoints)
    os.makedirs(save_root, exist_ok=True)
    path = os.path.join(save_root, "dataset_statistics.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[✓] dataset_statistics.json → {path}  (keys: {list(stats.keys())})")
    return path


# ---------------------------------------------------------------------
# Small save helpers with size reporting
# ---------------------------------------------------------------------
def save_dict_with_report(obj: Dict[str, torch.Tensor], path: str, label: str = "dict") -> None:
    """torch.save + print total bytes (element-size × numel)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path)
    total = sum(v.numel() * v.element_size() for v in obj.values() if torch.is_tensor(v))
    print(f"    [✓] saved {label}: {path}  ({total / (1024**3):.3f} GB, {len(obj)} tensors)")


def dir_size_gb(p: str) -> float:
    if not os.path.isdir(p):
        return 0.0
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024**3)


# ---------------------------------------------------------------------
# Optional: reload base model with new weights + save
# ---------------------------------------------------------------------
def save_as_hf_checkpoint(
    save_path: str,
    structure_ckpt: str,
    overlaid_sd: Dict[str, torch.Tensor],
    processor_ckpt: str = None,
    shared_dataset_stats: dict = None,
) -> None:
    """Save a full HF checkpoint where only `overlaid_sd` keys are overwritten."""
    os.makedirs(save_path, exist_ok=True)

    model = load_model(structure_ckpt)
    missing, unexpected = model.load_state_dict(overlaid_sd, strict=False)
    print(f"    [i] load_state_dict(strict=False): "
          f"missing={len(missing)} (expected: identical keys), unexpected={len(unexpected)}")
    if unexpected:
        print(f"    [!] unexpected keys sample: {unexpected[:3]}")
    model.save_pretrained(save_path)
    del model
    free_memory()

    p_src = processor_ckpt or structure_ckpt
    processor = AutoProcessor.from_pretrained(p_src, trust_remote_code=True)
    processor.save_pretrained(save_path)

    if shared_dataset_stats is not None:
        with open(os.path.join(save_path, "dataset_statistics.json"), "w") as f:
            json.dump(shared_dataset_stats, f, indent=2)

    print(f"    [✓] Saved HF checkpoint → {save_path}")
