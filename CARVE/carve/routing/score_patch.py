"""
Patch script — add score_orthonormal to concord_router_core.py.

Adds a new score function that:
1. Orthonormalizes Q (via SVD U) — drops sqrt(S) magnitude
2. Score = ||X @ Q_basis||² / ||X||²

This removes ||Q|| bias entirely while still measuring how much of X lives
in m's correction subspace.
"""
import sys
path = sys.argv[1]
src = open(path).read()

# Find score_q_only and add score_orthonormal right after it
new_func = '''

def score_orthonormal(
    activations: Dict[str, torch.Tensor],
    routing_keys_per_task: Dict[str, Dict[str, torch.Tensor]],
    *,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Orthonormal projection score.

    For each Q_m (n, r), compute orthonormal column basis B_m (n, r) via SVD,
    then score:
        s_m^(k) = ||X B_m||_F^2 / (||X||_F^2 + eps)
    Aggregate by mean over keys.

    This removes ||Q|| magnitude bias and measures how much of X lies in m's
    correction column space. Empirically discriminative on LIBERO bundles
    where Q factors share a common dominant direction.
    """
    # Pre-compute orthonormal bases (cache per call; small cost)
    bases_per_task: Dict[str, Dict[str, torch.Tensor]] = {}
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

    scores: Dict[str, float] = {}
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
            xnorm2 = Xf.pow(2).sum()
            denom = xnorm2 + eps
            total += float((num / denom).item())
            n_used += 1
        scores[task] = total / max(n_used, 1)
    return scores
'''

# Insert after score_q_only definition
anchor = '    return scores\n\n\ndef score_full_response('
if anchor not in src:
    print("ANCHOR NOT FOUND")
    sys.exit(1)

new_src = src.replace(anchor, '    return scores\n' + new_func + '\n\ndef score_full_response(')

# Update select_task to support new mode
old_select = '''    if score_mode == "q_only":
        raw = score_q_only(activations, routing_keys_per_task, eps=eps)
    elif score_mode == "full_response":
        if routing_keys_with_p is None:
            raise ValueError("score_mode='full_response' requires routing_keys_with_p")
        raw = score_full_response(activations, routing_keys_with_p, eps=eps)
    else:
        raise ValueError(f"unknown score_mode={score_mode}")'''

new_select = '''    if score_mode == "q_only":
        raw = score_q_only(activations, routing_keys_per_task, eps=eps)
    elif score_mode == "orthonormal":
        raw = score_orthonormal(activations, routing_keys_per_task, eps=eps)
    elif score_mode == "full_response":
        if routing_keys_with_p is None:
            raise ValueError("score_mode='full_response' requires routing_keys_with_p")
        raw = score_full_response(activations, routing_keys_with_p, eps=eps)
    else:
        raise ValueError(f"unknown score_mode={score_mode}")'''

new_src = new_src.replace(old_select, new_select)

open(path, "w").write(new_src)
print("patch applied")
