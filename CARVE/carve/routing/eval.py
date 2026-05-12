"""ConcordRouter V2 launcher (lazy routing variant).

This is the preferred entry point. It:
  1. Detects bundle from cfg.pretrained_checkpoint.
  2. Loads base VLA = theta_0.
  3. Stage A: applies tau_core only -> theta_shared.
  4. Installs a one-shot lazy router on model.forward. The first real forward
     call (made by the LIBERO eval rollout loop) captures activations,
     selects m_star, applies Stage B, and re-runs forward to produce the
     executed-model output. All subsequent calls go straight through.

This avoids the chicken-and-egg of needing a real observation before the eval
pipeline has reset the environment.

Override routing config via env vars:
  CONCORDROUTER_LAYERS=16,17,18,19,20
  CONCORDROUTER_WEIGHTS=self_attn.v_proj.weight,self_attn.o_proj.weight
  CONCORDROUTER_SCORE=q_only      # or full_response
"""
from __future__ import annotations

import argparse
import os
import sys
import runpy
from copy import copy

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SELF_DIR, os.environ.get("MERGEVLA_SRC", _SELF_DIR)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from carve.routing.overlay import (  # noqa: E402
    apply_core_only_inplace_,
    detect_v2_bundle,
    load_dataset_stats,
    load_merge_config,
)
from carve.routing.keys_io import (  # noqa: E402
    list_admitted_tasks,
    collect_routing_layer_keys,
)
from carve.routing.probe import install_lazy_router
from carve.routing.text_probe import install_text_lazy_router
from carve.routing.overlay import (
    apply_task_correction_inplace_with_track_,
    revert_delta_inplace_,
)
from transformers import AutoTokenizer  # noqa: E402


def _routing_layer_indices() -> list:
    raw = os.environ.get("CONCORDROUTER_LAYERS", "")
    if raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    return [16, 17, 18, 19, 20]


def _routing_weight_filter() -> list:
    raw = os.environ.get("CONCORDROUTER_WEIGHTS", "")
    if raw:
        return [w.strip() for w in raw.split(",") if w.strip()]
    return ["self_attn.v_proj.weight", "self_attn.o_proj.weight"]


def _score_mode() -> str:
    return os.environ.get("CONCORDROUTER_SCORE", "q_only")


def _patch_get_vla() -> None:
    import experiments.robot.openvla_utils as utils

    if getattr(utils.get_vla, "_concord_router_patched", False):
        return
    original_get_vla = utils.get_vla

    def _patched_get_vla(cfg):
        spec = detect_v2_bundle(cfg.pretrained_checkpoint)
        if spec is None:
            return original_get_vla(cfg)

        bundle_root, hint_task = spec
        merge_config = load_merge_config(bundle_root)
        base_ckpt = merge_config["base_checkpoint"]

        score_mode = _score_mode()
        layers = _routing_layer_indices()
        wfilter = _routing_weight_filter()

        print("=" * 72)
        print("[concord-router] V2 LAZY ROUTING")
        print(f"               bundle      = {bundle_root}")
        print(f"               path-hint   = {hint_task}  (NOT used)")
        print(f"               base        = {base_ckpt}")
        print(f"               score_mode  = {score_mode}")
        print(f"               layers      = {layers}")
        print(f"               weights     = {wfilter}")
        print("=" * 72)

        # 1. Base model
        base_cfg = copy(cfg)
        base_cfg.pretrained_checkpoint = base_ckpt
        model = original_get_vla(base_cfg)

        # 2. Stage A: tau_core only -> probe model
        apply_core_only_inplace_(model, bundle_root)

        # 3. Resolve routing config
        admitted_tasks = list_admitted_tasks(bundle_root)
        if not admitted_tasks:
            raise RuntimeError(f"no admitted tasks in {bundle_root}")
        ref_residual = os.path.join(
            bundle_root, admitted_tasks[0], "residual.safetensors"
        )
        base_keys = collect_routing_layer_keys(ref_residual, layers, wfilter)
        print(f"[concord-router] admitted: {admitted_tasks}")
        print(f"[concord-router] base_keys: {len(base_keys)} tensors")

        # 4. Install lazy router (fires on first forward)
        log_dir = os.path.join(bundle_root, "shared")

        if score_mode == "text":
            print("[concord-router] using TEXT routing mode")
            tokenizer = AutoTokenizer.from_pretrained(
                base_ckpt, trust_remote_code=True,
            )

            def apply_stage_b(m, task_name):
                _stat, delta = apply_task_correction_inplace_with_track_(
                    m, bundle_root, task_name, merge_config,
                )
                return delta

            def revert(m, delta):
                revert_delta_inplace_(m, delta)

            router = install_text_lazy_router(
                model=model,
                tokenizer=tokenizer,
                bundle_root=bundle_root,
                apply_stage_b_fn=apply_stage_b,
                revert_fn=revert,
                wrap_target="forward",
                log_dir=log_dir,
                use_template=True,
                layer=-1,
                pool="mean",
            )
        else:
            router = install_lazy_router(
                model=model,
                bundle_root=bundle_root,
                merge_config=merge_config,
                admitted_tasks=admitted_tasks,
                routing_layers=layers,
                weight_filter=wfilter,
                base_keys=base_keys,
                score_mode=score_mode,
                log_dir=log_dir,
            )
        # Attach to model for per-episode reset hook in eval loop
        model._concord_router_v2 = router

        # 5. norm_stats (V1 parity)
        try:
            ds_stats = load_dataset_stats(bundle_root)
            if hasattr(model, "norm_stats") and isinstance(model.norm_stats, dict):
                model.norm_stats.update(ds_stats)
            else:
                setattr(model, "norm_stats", ds_stats)
            print(f"[concord-router] norm_stats updated: {len(ds_stats)} entries")
        except Exception as e:
            print(f"[concord-router] WARN norm_stats: {type(e).__name__}: {e}")

        try:
            model.vla_path = cfg.pretrained_checkpoint
        except Exception:
            pass

        return model

    _patched_get_vla._concord_router_patched = True
    utils.get_vla = _patched_get_vla
    print("[concord-router] experiments.robot.openvla_utils.get_vla patched")


def _patch_get_processor() -> None:
    import experiments.robot.openvla_utils as utils
    if not hasattr(utils, "get_processor"):
        return
    if getattr(utils.get_processor, "_concord_router_patched", False):
        return
    original_get_processor = utils.get_processor

    def _patched_get_processor(cfg):
        spec = detect_v2_bundle(cfg.pretrained_checkpoint)
        if spec is None:
            return original_get_processor(cfg)
        bundle_root, _ = spec
        merge_config = load_merge_config(bundle_root)
        base_cfg = copy(cfg)
        base_cfg.pretrained_checkpoint = merge_config["base_checkpoint"]
        return original_get_processor(base_cfg)

    _patched_get_processor._concord_router_patched = True
    utils.get_processor = _patched_get_processor


def main() -> None:
    _patch_get_vla()
    _patch_get_processor()
    target = "experiments.robot.libero.run_libero_eval"
    print(f"[concord-router] handing off to {target}")
    runpy.run_module(target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
