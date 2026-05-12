"""Launcher: evaluate a CMM LR-Residual bundle task on LIBERO."""

from __future__ import annotations

import os
import sys
import runpy
from copy import copy

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SELF_DIR, os.environ.get("MERGEVLA_SRC", _SELF_DIR)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from carve.eval.overlay_oracle import (  # noqa: E402
    detect_lr_bundle,
    apply_lr_overlay_inplace_,
    load_lr_merge_config,
    load_lr_dataset_stats,
)


def _patch_get_vla() -> None:
    import experiments.robot.openvla_utils as utils

    if getattr(utils.get_vla, "_mergevla_lr_patched", False):
        return
    original_get_vla = utils.get_vla

    def _patched_get_vla(cfg):
        spec = detect_lr_bundle(cfg.pretrained_checkpoint)
        if spec is None:
            return original_get_vla(cfg)

        bundle_root, task_name = spec
        merge_config = load_lr_merge_config(bundle_root)
        base_ckpt = merge_config["base_checkpoint"]

        print("=" * 72)
        print("[lr overlay] CMM LR BUNDLE DETECTED")
        print(f"             bundle   = {bundle_root}")
        print(f"             task     = {task_name}")
        print(f"             scope    = {merge_config.get('scope')}")
        print(f"             gamma    = {merge_config.get('gamma')}")
        print(f"             rank_max = {merge_config.get('rank_max')}")
        print(f"             adapt    = {merge_config.get('rank_adaptive_threshold')}")
        print(f"             use_beta = {merge_config.get('use_beta')}")
        print(f"             base     = {base_ckpt}")
        print("=" * 72)

        base_cfg = copy(cfg)
        base_cfg.pretrained_checkpoint = base_ckpt
        model = original_get_vla(base_cfg)
        apply_lr_overlay_inplace_(model, bundle_root, task_name, merge_config)

        try:
            ds_stats = load_lr_dataset_stats(bundle_root)
            if hasattr(model, "norm_stats") and isinstance(model.norm_stats, dict):
                model.norm_stats.update(ds_stats)
            else:
                setattr(model, "norm_stats", ds_stats)
            print(f"[lr overlay] norm_stats updated with {len(ds_stats)} entries")
        except Exception as e:
            print(f"[lr overlay] WARN: norm_stats update failed: {type(e).__name__}: {e}")

        try:
            model.vla_path = cfg.pretrained_checkpoint
        except Exception:
            pass
        return model

    _patched_get_vla._mergevla_lr_patched = True  # type: ignore[attr-defined]
    utils.get_vla = _patched_get_vla
    print("[lr overlay] experiments.robot.openvla_utils.get_vla has been patched")


def _patch_get_processor() -> None:
    import experiments.robot.openvla_utils as utils

    if not hasattr(utils, "get_processor"):
        return
    if getattr(utils.get_processor, "_mergevla_lr_patched", False):
        return
    original_get_processor = utils.get_processor

    def _patched_get_processor(cfg):
        spec = detect_lr_bundle(cfg.pretrained_checkpoint)
        if spec is None:
            return original_get_processor(cfg)
        bundle_root, _ = spec
        merge_config = load_lr_merge_config(bundle_root)
        base_cfg = copy(cfg)
        base_cfg.pretrained_checkpoint = merge_config["base_checkpoint"]
        print(f"[lr overlay] get_processor redirected to base ({merge_config['base_checkpoint']})")
        return original_get_processor(base_cfg)

    _patched_get_processor._mergevla_lr_patched = True  # type: ignore[attr-defined]
    utils.get_processor = _patched_get_processor
    print("[lr overlay] experiments.robot.openvla_utils.get_processor has been patched")


def _locate_run_libero_eval() -> str:
    repo_root = os.environ.get("OPENVLA_REPO", os.getcwd())
    candidate = os.path.join(repo_root, "experiments", "robot", "libero", "run_libero_eval.py")
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"Could not find run_libero_eval.py at {candidate!r}. "
            "Set OPENVLA_REPO or run this launcher from the OpenVLA repo root."
        )
    return candidate


def main() -> None:
    _patch_get_vla()
    _patch_get_processor()
    target = _locate_run_libero_eval()
    print(f"[lr overlay] handing off to {target}")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
