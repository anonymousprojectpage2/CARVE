"""
CARVE — quickstart example.

Programmatic admit + oracle eval of a single LIBERO task.
For full pipeline, prefer the shell scripts under scripts/.
"""

from pathlib import Path

# Admit one task
from carve.admit.merge import admit_task

bundle_root = Path("./carve_demo_bundle")
admit_task(
    bundle_root=bundle_root,
    task_name="libero_spatial",
    task_ckpt="openvla/openvla-7b-finetuned-libero-spatial",
    base_ckpt="openvla/openvla-7b",
    rank_max=64,
)

# Evaluate with the oracle overlay
from carve.eval.oracle import evaluate

evaluate(
    bundle=bundle_root / "libero_spatial",
    task_suite_name="libero_spatial",
    num_trials_per_task=2,
)
