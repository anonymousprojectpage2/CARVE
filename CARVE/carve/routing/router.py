"""
ConcordRouter V2 — core routing logic.

Single shared-model probe + correction-response scoring.

Pipeline:
  1. Build  theta_shared = theta_0 + C_T   (V1 admit, then ignore residuals).
  2. Forward (o_0, q) once on theta_shared, capture activations X^(k) at routing
     layers via forward_pre_hooks on Linear modules.
  3. For each candidate task m, score using stored Q_m^(k):
         s_m^(k) = ||X Q||_F^2 / (||X||_F^2 ||Q||_F^2 + eps)        # main
         s_m     = mean over k in R of s_m^(k)
  4. Argmax over m, instantiate full theta_exec for that task.

This module is model-agnostic — `theta_shared` is any nn.Module containing
nn.Linear modules whose qualified names match the residual base keys.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Activation capture
# ----------------------------------------------------------------------------

class _ActivationCatcher:
    """Capture inputs to selected nn.Linear modules during a single forward.

    Routing keys (base_keys) are formatted as `{module_qualified_name}.weight`,
    e.g. `language_model.model.layers.16.self_attn.v_proj.weight`. We strip
    `.weight` to find the matching nn.Module via model.get_submodule(...).
    """

    def __init__(self, model: nn.Module, base_keys: List[str]):
        self.model = model
        self.base_keys = base_keys
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _module_name_for(self, base_key: str) -> str:
        # base_key is e.g. "language_model.model.layers.16.self_attn.v_proj.weight"
        if base_key.endswith(".weight"):
            return base_key[: -len(".weight")]
        return base_key

    def __enter__(self):
        for bk in self.base_keys:
            mod_name = self._module_name_for(bk)
            try:
                mod = self.model.get_submodule(mod_name)
            except AttributeError:
                # Skip silently — caller will see missing entry in self.activations
                continue
            if not isinstance(mod, nn.Linear):
                continue

            def make_hook(key):
                def hook(_module, inputs):
                    # inputs is a tuple; the first arg is the input activation
                    x = inputs[0]
                    if not torch.is_tensor(x):
                        return
                    # Detach + flatten leading dims => (T, in_features)
                    x_flat = x.reshape(-1, x.shape[-1]).detach()
                    self.activations[key] = x_flat
                return hook

            h = mod.register_forward_pre_hook(make_hook(bk))
            self._handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def capture_routing_activations(
    model: nn.Module,
    routing_base_keys: List[str],
    forward_fn,
) -> Dict[str, torch.Tensor]:
    """Run forward_fn(model) once with hooks on routing keys.

    Args:
        model: theta_shared (theta_0 + C_T loaded).
        routing_base_keys: residual base keys (e.g. "...v_proj.weight").
        forward_fn: callable taking `model` as its only arg, performing one
                    forward pass on the initial observation/instruction.

    Returns:
        Dict mapping base_key -> activation tensor, shape (T, in_features).
    """
    with _ActivationCatcher(model, routing_base_keys) as cat:
        with torch.no_grad():
            forward_fn(model)
        return dict(cat.activations)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def score_q_only(
    activations: Dict[str, torch.Tensor],
    routing_keys_per_task: Dict[str, Dict[str, torch.Tensor]],
    *,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Q-only input-subspace routing score (main).

    For each task m, key k in routing layers:
        s_m^(k) = ||X Q||_F^2 / (||X||_F^2 ||Q||_F^2 + eps)
    Aggregate by mean over keys.

    Args:
        activations: base_key -> X, shape (T, in_features).
        routing_keys_per_task: task_name -> {base_key -> Q, shape (in, r)}.

    Returns:
        Dict task_name -> score (float).
    """
    scores: Dict[str, float] = {}
    for task, keys_for_task in routing_keys_per_task.items():
        total = 0.0
        n_used = 0
        for bk, Q in keys_for_task.items():
            X = activations.get(bk, None)
            if X is None:
                continue
            # Move Q to same device/dtype as X for matmul, but compute in fp32.
            Xf = X.float()
            Qf = Q.to(device=Xf.device, dtype=torch.float32)
            # Shape check: X is (T, in), Q is (in, r) -> X @ Q is (T, r)
            if Xf.shape[-1] != Qf.shape[0]:
                continue
            num = (Xf @ Qf).pow(2).sum()
            xnorm2 = Xf.pow(2).sum()
            qnorm2 = Qf.pow(2).sum()
            denom = xnorm2 * qnorm2 + eps
            total += float((num / denom).item())
            n_used += 1
        scores[task] = total / max(n_used, 1)
    return scores


def score_orthonormal(
    activations,
    routing_keys_per_task,
    *,
    eps: float = 1e-8,
):
    """Orthonormal projection score (V2 main score function).

    For each Q_m (n, r), compute orthonormal column basis B_m (n, r) via SVD,
    then score:
        s_m^(k) = ||X B_m||_F^2 / (||X||_F^2 + eps)
    Aggregate by mean over keys.
    """
    bases_per_task = {}
    for task, keys_for_task in routing_keys_per_task.items():
        bases_per_task[task] = {}
        for bk, Q in keys_for_task.items():
            Qf = Q.float()
            try:
                U, _, _ = torch.svd(Qf)
            except RuntimeError:
                continue
            r = min(Qf.shape[1], U.shape[1])
            bases_per_task[task][bk] = U[:, :r].contiguous()

    scores = {}
    for task, bases in bases_per_task.items():
        total = 0.0
        n_used = 0
        for bk, B in bases.items():
            X = activations.get(bk, None)
            if X is None:
                continue
            Xf = X.float()
            Bf = B.to(device=Xf.device, dtype=torch.float32)
            if Xf.shape[-1] != Bf.shape[0]:
                continue
            num = (Xf @ Bf).pow(2).sum()
            denom = Xf.pow(2).sum() + eps
            total += float((num / denom).item())
            n_used += 1
        scores[task] = total / max(n_used, 1)
    return scores


def score_text(
    instruction: str,
    text_keys: dict,
    *,
    tokenizer,
    model,
    layer: int = -1,
    pool: str = "mean",
    use_template: bool = True,
) -> dict:
    """Score routing using base VLM's text encoder.
    
    Returns: {task_name: cosine_sim} for each admitted task
    """
    import torch
    if use_template:
        text = f"In: What action should the robot take to {instruction.lower()}?\nOut: \u2581"
    else:
        text = instruction
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.language_model(ids, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[layer]  # (1, L, d)
    if pool == "mean":
        emb = hs.mean(dim=1).squeeze(0)
    else:
        emb = hs[:, -1, :].squeeze(0)
    emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu()
    
    scores = {}
    for task_name, key in text_keys.items():
        # text_keys are already L2 normalized
        cos = (emb * key).sum().item()
        scores[task_name] = float(cos)
    return scores


def score_full_response(
    activations: Dict[str, torch.Tensor],
    routing_keys_with_p: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    *,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Full-response routing score (ablation).

    s_m^(k) = ||X Q P^T||_F^2 / (||X||_F^2 ||P Q^T||_F^2 + eps)

    Args:
        activations: base_key -> X.
        routing_keys_with_p: task -> {base_key -> (P, Q)}, P (m, r), Q (n, r).
    """
    scores: Dict[str, float] = {}
    for task, keys_for_task in routing_keys_with_p.items():
        total = 0.0
        n_used = 0
        for bk, (P, Q) in keys_for_task.items():
            X = activations.get(bk, None)
            if X is None:
                continue
            Xf = X.float()
            Pf = P.to(device=Xf.device, dtype=torch.float32)
            Qf = Q.to(device=Xf.device, dtype=torch.float32)
            if Xf.shape[-1] != Qf.shape[0]:
                continue
            # X Q P^T  -> (T, m)
            xq = Xf @ Qf                # (T, r)
            xqpt = xq @ Pf.t()          # (T, m)
            num = xqpt.pow(2).sum()
            xnorm2 = Xf.pow(2).sum()
            # ||P Q^T||_F^2 = ||P||_F^2 * ||Q||_F^2  if cols are orthonormal,
            # but P, Q have sqrt(S) baked in -> compute exactly:
            pqt_norm2 = (Pf @ Qf.t()).pow(2).sum()
            denom = xnorm2 * pqt_norm2 + eps
            total += float((num / denom).item())
            n_used += 1
        scores[task] = total / max(n_used, 1)
    return scores


# ----------------------------------------------------------------------------
# Top-level routing
# ----------------------------------------------------------------------------

def softmax_dict(scores: Dict[str, float]) -> Dict[str, float]:
    keys = list(scores.keys())
    vals = torch.tensor([scores[k] for k in keys], dtype=torch.float32)
    probs = torch.softmax(vals, dim=0).tolist()
    return {k: float(p) for k, p in zip(keys, probs)}


def select_task(
    activations: Dict[str, torch.Tensor],
    routing_keys_per_task: Dict[str, Dict[str, torch.Tensor]],
    *,
    score_mode: str = "q_only",
    routing_keys_with_p: Optional[
        Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]
    ] = None,
    eps: float = 1e-8,
    verbose: bool = True,
    log_prefix: str = "[concord-router]",
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """Compute scores, return (m_star, raw_scores, softmax_probs)."""
    if score_mode == "q_only":
        raw = score_q_only(activations, routing_keys_per_task, eps=eps)
    elif score_mode == "orthonormal":
        raw = score_orthonormal(activations, routing_keys_per_task, eps=eps)
    elif score_mode == "text":
        # text mode handled separately by router; this is fallback
        raise ValueError("score_mode='text' must be handled in the router probe, not select_task()")
    elif score_mode == "full_response":
        if routing_keys_with_p is None:
            raise ValueError("score_mode='full_response' requires routing_keys_with_p")
        raw = score_full_response(activations, routing_keys_with_p, eps=eps)
    else:
        raise ValueError(f"unknown score_mode={score_mode}")

    probs = softmax_dict(raw)
    m_star = max(probs, key=probs.get)

    if verbose:
        print(f"{log_prefix} score_mode={score_mode}")
        for m in sorted(raw.keys()):
            print(f"{log_prefix}   {m:<32s}  raw={raw[m]:.6f}  prob={probs[m]:.4f}")
        print(f"{log_prefix} selected: {m_star}")

    return m_star, raw, probs
