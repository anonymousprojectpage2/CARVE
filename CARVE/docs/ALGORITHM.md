# The CARVE Algorithm

## Setup

We are given a frozen pretrained VLA model with weights $\theta_0$, and a
stream of task-specific fine-tuned experts $\theta_1, \theta_2, \dots,
\theta_T$ arriving one at a time. The goal is to support all $T$ tasks at
inference time with storage cost much smaller than naively keeping $T$
independent copies of the model.

We define the per-task **task vector**:

$$
\tau_t = \theta_t - \theta_0
$$

We maintain a single shared **core delta** $\tau_{\text{core}}$, initialized
to zero, and a per-task **residual** $r_t$, both kept on disk in compressed
form.

## Admit

When task $t$ arrives, we update the shared core under a **sign-protection**
rule and then save what the core could not absorb as the task's residual.

For each coordinate $i$:

1. **Sign protection.** Coordinate $i$ is *protected* iff
   $\tau_{\text{core},i} = 0$ or
   $\operatorname{sign}(\tau_{\text{core},i}) = \operatorname{sign}(\tau_{t,i})$.

2. **EMA on protected coordinates only.**

   $$
   \tau_{\text{core},i}^{\text{new}} =
   \begin{cases}
     (1-\alpha_t)\,\tau_{\text{core},i}^{\text{old}} + \alpha_t \tau_{t,i}
       & \text{if protected} \\[2pt]
     \tau_{\text{core},i}^{\text{old}}
       & \text{otherwise}
   \end{cases}
   $$

   with $\alpha_t = 1/\sqrt{t}$. This step is performed independently per
   "scope group" (`per_block` by default) so the update rate adapts to where
   in the model the conflict appears.

3. **Residual.**

   $$
   r_t = \tau_t - \tau_{\text{core}}^{\text{new}}
   $$

   Per tensor key $k$: if $r_t[k]$ is 1D or has fewer than
   `min_factorise_numel` entries, store the raw bf16 vector. Otherwise apply
   randomized SVD with rank $r$:

   $$
   r_t[k] \approx U_t[k] \, V_t[k]^\top,
   \quad U_t[k] \in \mathbb{R}^{m \times r},\;
         V_t[k] \in \mathbb{R}^{n \times r}.
   $$

The residual rank $r$ controls the storage/quality trade-off (see ablation in
the paper).

## Oracle overlay

At inference for task $m$, if the task identity is known, we reconstruct
weights as:

$$
\theta_m = \theta_0 + \beta\,\tau_{\text{core}} + r_m
$$

where $r_m[k] = U_m[k] V_m[k]^\top$ for factorised keys and the raw bf16
vector otherwise. $\beta$ is a learned (or default 1.0) overlay scale.

This is the **`carve.eval.oracle`** entry point.

## Routing overlay

When the task identity is *not* known at inference, we instead use the
**routing** module. After admitting all $T$ tasks we run a one-shot
key-building pass that produces a small per-task signature
$\kappa_t \in \mathbb{R}^d$ (`carve.routing.keys`).

At inference, given the model's input context, we compute a query vector
$q$ in the same space and pick the top-$K$ tasks by cosine similarity. The
overlay then becomes a softmax mixture over the selected residuals:

$$
\theta_m = \theta_0 + \beta\,\tau_{\text{core}} + \sum_{k \in \text{top-}K} w_k\,r_k,
\qquad w = \operatorname{softmax}\bigl(\{\langle q, \kappa_k \rangle\}_{k}\bigr).
$$

This is the **`carve.routing.eval`** entry point. The routing keys are tiny
(a few hundred KB per task), so the additional storage cost is negligible.

## Memory discipline

- $\tau_{\text{core}}$ is held in bf16 both in memory and on disk.
- During admit, we materialize fp32 working copies of base, task, core,
  $\tau_t$, and residual *per tensor key*, and free them before moving on.
- The randomized SVD step uses `torch.svd_lowrank`, which is cheap even for
  large LLM weight matrices like 4096×11008.

## Storage accounting

For a model with $P$ pretrained parameters and $T$ admitted tasks at rank
$r$, the on-disk state is:

$$
\underbrace{P}_{\text{pretrained, shared}}
\;+\;
\underbrace{P}_{\tau_{\text{core}}}
\;+\;
T \cdot \underbrace{r \cdot (\text{sum of } m_k + n_k)}_{\text{LR residuals}}
$$

For OpenVLA-7B with $T=4$ and $r=64$ this works out to ~15.95 B parameters,
versus 30.16 B for the naive 4-copies baseline — a 47% reduction.

See the paper for the full rank ablation table.
