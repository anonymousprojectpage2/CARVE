"""
ConcordRouter V2 — lazy routing helper.

LIBERO eval calls get_vla(cfg) BEFORE the environment is reset, so we don't
have a real initial observation when the model is built. We solve this by
deferring routing to the first real forward pass:

  1. After Stage A in run_eval_v2._patched_get_vla, we wrap model.forward so
     that on the FIRST call, we capture activations, score corrections, decide
     m_star, run Stage B in-place, then re-execute the wrapped forward so the
     caller gets the executed-model output.
  2. All subsequent forward calls go through the unwrapped (now V1) model
     directly.

Public entry point: install_lazy_router(model, bundle_root, base_keys, ...).
The earlier `run_probe_forward(...)` reference in run_eval_v2 is replaced by
this install function.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from carve.routing.router import _ActivationCatcher, select_task  # noqa: E402
from carve.routing.overlay import (  # noqa: E402
    apply_task_correction_inplace_,
    apply_task_correction_inplace_with_track_,
    revert_delta_inplace_,
)
from carve.routing.keys_io import (  # noqa: E402
    load_all_tasks_routing_keys,
    load_routing_keys_with_p,
)


class _LazyRouter:
    """One-shot routing wrapper. Replaces model.forward until first call."""

    def __init__(
        self,
        model: nn.Module,
        bundle_root: str,
        merge_config: dict,
        admitted_tasks: List[str],
        routing_layers: List[int],
        weight_filter: List[str],
        base_keys: List[str],
        score_mode: str = "q_only",
        log_dir: Optional[str] = None,
    ):
        self.model = model
        self.bundle_root = bundle_root
        self.merge_config = merge_config
        self.admitted_tasks = admitted_tasks
        self.routing_layers = routing_layers
        self.weight_filter = weight_filter
        self.base_keys = base_keys
        self.score_mode = score_mode
        self.log_dir = log_dir
        self._fired = False
        self._stage_b_delta = None  # CPU fp32 dict for per-episode revert
        self._original_forward = model.forward

        # Pre-load routing keys (cheap, ~few MB total)
        if score_mode in ("q_only", "orthonormal"):
            self.rk_per_task = load_all_tasks_routing_keys(
                bundle_root, admitted_tasks, routing_layers, weight_filter,
                dtype=torch.float32,
            )
            self.rk_with_p = None
        elif score_mode == "full_response":
            self.rk_with_p = {
                m: load_routing_keys_with_p(
                    bundle_root, m, routing_layers, weight_filter,
                    dtype=torch.float32,
                )
                for m in admitted_tasks
            }
            self.rk_per_task = {
                m: {k: v for k, (u, v) in d.items()}
                for m, d in self.rk_with_p.items()
            }
        else:
            raise ValueError(f"unknown score_mode={score_mode}")

    def install(self):
        # Replace forward bound method
        self_ref = self
        original = self._original_forward

        def wrapped(*args, **kwargs):
            if self_ref._fired:
                return original(*args, **kwargs)
            return self_ref._first_call(args, kwargs, original)

        # bind on the instance
        self.model.forward = wrapped
        return self

    def _first_call(self, args, kwargs, original):
        print("[concord-router] LAZY ROUTING — first forward call detected")

        # 1) Capture activations during this forward
        with _ActivationCatcher(self.model, self.base_keys) as cat:
            with torch.no_grad():
                # Run probe forward; we do NOT need the output, but we must run
                # the same call so hooks capture realistic activations.
                # However, the caller wants the output too. Strategy:
                #   first run probe (eager) to capture activations
                #   then perform routing
                #   then run actual forward and return
                # The probe IS the actual forward — we just capture, then rerun
                # post-Stage-B for the executed model.
                _ = original(*args, **kwargs)
            activations = dict(cat.activations)

        print(f"[concord-router] captured {len(activations)} activations")
        for bk in sorted(activations.keys())[:3]:
            x = activations[bk]
            print(f"                  {bk[:60]}  shape={tuple(x.shape)}")

        # 2) Score and select
        m_star, raw, probs = select_task(
            activations,
            self.rk_per_task,
            score_mode=self.score_mode,
            routing_keys_with_p=self.rk_with_p,
        )

        # 3) Log decision
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "router_log.jsonl"), "a") as f:
                f.write(json.dumps({
                    "selected": m_star,
                    "scores": raw,
                    "probs": probs,
                    "layers": self.routing_layers,
                    "weights": self.weight_filter,
                    "score_mode": self.score_mode,
                }) + "\n")

        # 4) Apply Stage B for selected task (with delta tracking for reset)
        _stat, self._stage_b_delta = apply_task_correction_inplace_with_track_(
            self.model, self.bundle_root, m_star, self.merge_config,
        )
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5) Restore original forward and re-run to produce executed-model output
        self.model.forward = self._original_forward
        self._fired = True
        with torch.no_grad():
            return self._original_forward(*args, **kwargs)


    def reset_for_new_episode(self) -> None:
        """Per-episode reset: revert Stage B delta and re-arm router.

        After this returns, the next forward call will trigger routing again.
        Model state is restored to the probe state (theta_0 + C_T).
        Safe to call even if router has not fired (no-op).
        """
        if self._stage_b_delta is not None:
            revert_delta_inplace_(self.model, self._stage_b_delta,
                                  verbose=False)
            self._stage_b_delta = None
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._fired = False
        # Re-install wrapper
        self_ref = self
        original = self._original_forward

        def wrapped(*args, **kwargs):
            if self_ref._fired:
                return original(*args, **kwargs)
            return self_ref._first_call(args, kwargs, original)
        self.model.forward = wrapped
        # Quiet log on first reset, otherwise no-op verbose
        # (LIBERO eval prints "Starting episode" already; avoid duplicating)


def install_lazy_router(
    model: nn.Module,
    bundle_root: str,
    merge_config: dict,
    admitted_tasks: List[str],
    routing_layers: List[int],
    weight_filter: List[str],
    base_keys: List[str],
    *,
    score_mode: str = "q_only",
    log_dir: Optional[str] = None,
) -> _LazyRouter:
    """Attach lazy router to model.forward. Routing fires on first call."""
    router = _LazyRouter(
        model=model,
        bundle_root=bundle_root,
        merge_config=merge_config,
        admitted_tasks=admitted_tasks,
        routing_layers=routing_layers,
        weight_filter=weight_filter,
        base_keys=base_keys,
        score_mode=score_mode,
        log_dir=log_dir,
    )
    router.install()
    print(f"[concord-router] lazy router installed; will fire on first forward")
    return router


def run_probe_forward(model, base_keys, cfg):
    """Stub kept for backwards compatibility with run_eval_v2.

    Real routing is now lazy — done on first forward. This function returns
    empty dict so the caller in run_eval_v2 doesn't break, but the routing
    decision is deferred.

    NOTE: run_eval_v2 should be updated to call install_lazy_router instead
    of running probe + select_task eagerly. See run_eval_v2_lazy.py for the
    cleaner integration.
    """
    print("[concord-router] eager run_probe_forward called — returning empty "
          "(use install_lazy_router for real routing)")
    return {}
