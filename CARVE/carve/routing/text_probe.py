"""
Text-based ConcordRouter (Design E).

Uses the base VLM's language model to encode the current episode's instruction
and routes by cosine similarity against per-task mean instruction embeddings
saved in `text_routing_keys.pt`.

This is a separate module from carve.routing.probe.py to keep the two routing
modes (Q-factor vs text-embedding) cleanly distinguishable. They share the
overlay loader (Stage A/B) and the per-episode reset machinery.

Lazy semantics:
  - First call (per episode) extracts the instruction from input_ids and runs
    the text encoder, then applies Stage B for the chosen task.
  - Subsequent calls within the same episode pass through.
  - reset_for_new_episode() reverts Stage B and re-arms.

Instruction extraction:
  We don't have direct access to the raw instruction at predict_action time;
  what we have is `input_ids`. The OpenVLA prompt format is:
    "In: What action should the robot take to <instruction>?\nOut: ▁"
  We decode input_ids and parse out the substring between "to " and "?".
  If parsing fails, we fall back to the full decoded text.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn


def _extract_instruction_from_ids(input_ids, tokenizer) -> str:
    """Decode input_ids and extract the instruction substring.

    Returns the cleaned instruction string. Falls back to full decoded
    text if the OpenVLA prompt template is not detected.
    """
    if input_ids.dim() == 2:
        ids = input_ids[0]
    else:
        ids = input_ids
    text = tokenizer.decode(ids, skip_special_tokens=True)
    # Try OpenVLA template: "What action should the robot take to <inst>?"
    m = re.search(r"to (.+?)\?", text)
    if m:
        return m.group(1).strip()
    # Fallback: just return the full text
    return text.strip()


@torch.no_grad()
def encode_instruction(
    instruction: str,
    model,
    tokenizer,
    *,
    layer: int = -1,
    pool: str = "mean",
    use_template: bool = True,
) -> torch.Tensor:
    """Encode a single instruction using the base VLM's language model."""
    if use_template:
        text = f"In: What action should the robot take to {instruction.lower()}?\nOut: \u2581"
    else:
        text = instruction
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    out = model.language_model(ids, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[layer]  # (1, L, d)
    if pool == "mean":
        emb = hs.mean(dim=1).squeeze(0)
    elif pool == "last":
        emb = hs[:, -1, :].squeeze(0)
    else:
        raise ValueError(f"unknown pool={pool}")
    return torch.nn.functional.normalize(emb.float(), dim=-1).cpu()


def score_against_keys(
    emb: torch.Tensor,
    text_keys: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Cosine similarity (assuming both already normalized)."""
    scores = {}
    for task, key in text_keys.items():
        scores[task] = float((emb * key).sum().item())
    return scores


class _TextLazyRouter:
    """Per-episode text-based lazy router.

    Args:
      model: base VLM (already in probe state, theta_0 + C_T)
      tokenizer: paired tokenizer
      bundle_root: bundle directory containing text_routing_keys.pt
      apply_stage_b_fn: callable(model, task_name) -> delta_dict
                       (caller wraps overlay_loader_v2's _with_track_ variant)
      revert_fn: callable(model, delta_dict)
      wrap_target: 'forward' (OpenVLA) or 'predict_action' (MergeVLA)
      log_dir: optional logging directory
      use_template: whether to apply OpenVLA prompt template
      layer/pool: hidden state extraction
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        bundle_root: str,
        apply_stage_b_fn,
        revert_fn,
        *,
        wrap_target: str = "forward",
        log_dir: Optional[str] = None,
        use_template: bool = True,
        layer: int = -1,
        pool: str = "mean",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.bundle_root = bundle_root
        self.apply_stage_b_fn = apply_stage_b_fn
        self.revert_fn = revert_fn
        self.wrap_target = wrap_target
        self.log_dir = log_dir
        self.use_template = use_template
        self.layer = layer
        self.pool = pool

        # Load text routing keys (env var override for K-ablation)
        _override = os.environ.get("CONCORDROUTER_TEXT_KEYS_PATH", "")
        if _override:
            path = _override
        else:
            path = os.path.join(bundle_root, "text_routing_keys.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"text_routing_keys.pt not found at {path}. "
                "Run scripts/build_text_routing_keys.py first."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.text_keys = payload["text_keys"]  # {task: tensor (d,)}
        self.config = payload.get("config", {})
        # Override use_template / pool from saved config if available
        if "use_template" in self.config:
            self.use_template = self.config["use_template"]
        if "pool" in self.config:
            self.pool = self.config["pool"]

        self._fired = False
        self._stage_b_delta = None
        self._original_call = getattr(model, wrap_target)

        print(f"[concord-text-router] loaded {len(self.text_keys)} task keys "
              f"from {path}")
        print(f"[concord-text-router] tasks: {list(self.text_keys.keys())}")
        print(f"[concord-text-router] use_template={self.use_template}, pool={self.pool}")

    def install(self):
        self_ref = self
        original = self._original_call

        def wrapped(*args, **kwargs):
            if self_ref._fired:
                return original(*args, **kwargs)
            return self_ref._first_call(args, kwargs, original)

        setattr(self.model, self.wrap_target, wrapped)
        return self

    def _first_call(self, args, kwargs, original):
        # Extract input_ids from args/kwargs
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            # Try first positional arg
            for a in args:
                if torch.is_tensor(a) and a.dtype in (torch.long, torch.int64, torch.int32):
                    input_ids = a
                    break
        if input_ids is None:
            print("[concord-text-router] WARN: could not find input_ids; routing skipped")
            self._fired = True
            setattr(self.model, self.wrap_target, original)
            return original(*args, **kwargs)

        # Decode and extract instruction
        instruction = _extract_instruction_from_ids(input_ids, self.tokenizer)
        print(f"[concord-text-router] LAZY ROUTING — instruction: {instruction[:80]!r}")

        # Encode current instruction
        emb = encode_instruction(
            instruction, self.model, self.tokenizer,
            layer=self.layer, pool=self.pool,
            use_template=self.use_template,
        )

        # Score
        scores = score_against_keys(emb, self.text_keys)
        m_star = max(scores, key=scores.get)
        # Compute margin
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0

        print(f"[concord-text-router] scores:")
        for t, s in sorted(scores.items(), key=lambda x: -x[1]):
            mark = " ★" if t == m_star else ""
            print(f"  {t:<22s}  {s:.4f}{mark}")
        print(f"[concord-text-router] selected: {m_star}  (margin={margin:.4f})")

        # Log
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "router_log_text.jsonl"), "a") as f:
                f.write(json.dumps({
                    "instruction": instruction,
                    "selected": m_star,
                    "scores": scores,
                    "margin": margin,
                }) + "\n")

        # Stage B
        self._stage_b_delta = self.apply_stage_b_fn(self.model, m_star)
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Restore original, re-run
        setattr(self.model, self.wrap_target, self._original_call)
        self._fired = True
        with torch.no_grad():
            return self._original_call(*args, **kwargs)

    def reset_for_new_episode(self):
        if self._stage_b_delta is not None:
            self.revert_fn(self.model, self._stage_b_delta)
            self._stage_b_delta = None
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._fired = False
        self_ref = self
        original = self._original_call

        def wrapped(*args, **kwargs):
            if self_ref._fired:
                return original(*args, **kwargs)
            return self_ref._first_call(args, kwargs, original)

        setattr(self.model, self.wrap_target, wrapped)


def install_text_lazy_router(
    model: nn.Module,
    tokenizer,
    bundle_root: str,
    apply_stage_b_fn,
    revert_fn,
    *,
    wrap_target: str = "forward",
    log_dir: Optional[str] = None,
    use_template: bool = True,
    layer: int = -1,
    pool: str = "mean",
) -> _TextLazyRouter:
    """Install the text-based lazy router on `model`.

    `apply_stage_b_fn(model, task_name) -> delta_dict` and
    `revert_fn(model, delta_dict)` are wrappers around overlay_loader_v2's
    apply_task_correction_inplace_with_track_ and revert_delta_inplace_,
    respectively. The router does not couple to the bundle format; the
    caller adapts these for OpenVLA / MergeVLA.
    """
    router = _TextLazyRouter(
        model=model,
        tokenizer=tokenizer,
        bundle_root=bundle_root,
        apply_stage_b_fn=apply_stage_b_fn,
        revert_fn=revert_fn,
        wrap_target=wrap_target,
        log_dir=log_dir,
        use_template=use_template,
        layer=layer,
        pool=pool,
    )
    router.install()
    print(f"[concord-text-router] installed; will fire on first {wrap_target} call")
    return router
