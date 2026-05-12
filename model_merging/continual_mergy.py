import os
import json
import time
import copy
import torch
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

@dataclass
class ContinualMergeState:
    task_count: int = 0
    task_names: List[str] = field(default_factory=list)
    tau_sum: Dict[str, torch.Tensor] = field(default_factory=dict) 
    identical_keys: List[str] = field(default_factory=list)
    proprio_sum: Dict[str, torch.Tensor] = field(default_factory=dict)
    action_head_sum: Dict[str, torch.Tensor] = field(default_factory=dict)

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "task_count":     self.task_count,
            "task_names":     self.task_names,
            "tau_sum":        self.tau_sum,
            "identical_keys": self.identical_keys,
            "proprio_sum": self.proprio_sum,
            "action_head_sum": self.action_head_sum,
        }, path)
        print(f"[State] Saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ContinualMergeState":
        path = Path(path)
        d = torch.load(path, map_location="cpu")
        state = cls(**d)
        print(f"[State] Loaded from {path}  (task_count={state.task_count})")
        return state

def load_model(
    checkpoint: Union[str, Path],
    trust_remote_code: bool = True,
    torch_dtype=torch.bfloat16,
    device: str = "cpu",
) -> AutoModelForVision2Seq:
  
    checkpoint = str(checkpoint)
    print(f"  Loading model: {checkpoint}")
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    return model


def load_processor(checkpoint: Union[str, Path], trust_remote_code: bool = True):
    checkpoint = str(checkpoint)
    try:
        return AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=trust_remote_code
        )
    except Exception:
        return AutoImageProcessor.from_pretrained(
            checkpoint, trust_remote_code=trust_remote_code
        )

def compute_task_vector(
    pretrained_sd: Dict[str, torch.Tensor],
    task_sd: Dict[str, torch.Tensor],
    exclude_keys: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:

    exclude_keys = set(exclude_keys or [])
    tau = {}
    for k, v_pretrained in pretrained_sd.items():
        if k in exclude_keys:
            continue
        if k not in task_sd:
            continue
        v_task = task_sd[k].to(device)
        v_pre  = v_pretrained.to(device)
        if v_task.shape != v_pre.shape:
            print(f"  [Skip] shape mismatch: {k}  {v_pre.shape} vs {v_task.shape}")
            continue
        tau[k] = v_task - v_pre
    return tau


def find_identical_keys(
    pretrained_sd: Dict[str, torch.Tensor],
    task_sd:       Dict[str, torch.Tensor],
) -> List[str]:
    identical = []
    for k, v in pretrained_sd.items():
        if k not in task_sd:
            continue
        vt = task_sd[k]
        if v.shape == vt.shape and v.dtype == vt.dtype and torch.equal(v.cpu(), vt.cpu()):
            identical.append(k)
    return identical

def merge_step(
    task_name:        str,
    task_checkpoint:  Union[str, Path],
    base_checkpoint:  Union[str, Path],
    state:            Optional[ContinualMergeState] = None, 
    device:           str = "cpu",
    trust_remote_code: bool = True,
    torch_dtype:      torch.dtype = torch.bfloat16,
) -> ContinualMergeState:

    if state is None:
        state = ContinualMergeState()
    t = state.task_count + 1
    print(f"\n{'='*60}")
    print(f" STEP {t}: merging task '{task_name}'")
    print(f"{'='*60}")

    print(f"[Step {t}] Loading base model...")
    base_model = load_model(
        base_checkpoint,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device=device,
    )
    pretrained_sd = {k: v.clone().cpu() for k, v in base_model.state_dict().items()}
    del base_model
    torch.cuda.empty_cache()

    print(f"[Step {t}] Loading task model...")
    task_model = load_model(
        task_checkpoint,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device=device,
    )
    task_sd = {k: v.clone().cpu() for k, v in task_model.state_dict().items()}
    del task_model
    torch.cuda.empty_cache()

    if t == 1:
        print(f"[Step {t}] Computing identical_keys...")
        state.identical_keys = find_identical_keys(pretrained_sd, task_sd)
        print(f"[Step {t}] identical_keys: {len(state.identical_keys)} params (excluded from τ)")
    else:
        removed = [
            k for k in state.identical_keys
            if k in task_sd and not torch.equal(pretrained_sd[k].cpu(), task_sd[k].cpu())
        ]
        if removed:
            print(f"[Step {t}] Removing {len(removed)} keys from identical_keys.")
            state.identical_keys = [k for k in state.identical_keys if k not in removed]

    print(f"[Step {t}] Computing task vector τ_t...")
    tau_t = compute_task_vector(
        pretrained_sd, task_sd,
        exclude_keys=state.identical_keys,
        device="cpu",
    )
    print(f"[Step {t}] τ_t keys: {len(tau_t)}")

    for k, delta in tau_t.items():
        if k in state.tau_sum:
            state.tau_sum[k] = state.tau_sum[k] + delta
        else:
            state.tau_sum[k] = delta.clone()

    state.task_count += 1
    state.task_names.append(task_name)

    print(f"[Step {t}] Loading proprio_projector and action_head...")
    for fname, sum_attr in [("proprio_projector--checkpoint.pt", "proprio_sum"),
                             ("action_head--checkpoint.pt", "action_head_sum")]:
        src = Path(str(task_checkpoint)) / fname
        sd = None
        if src.exists():
            sd = torch.load(src, map_location="cpu")
        else:
            try:
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(repo_id=str(task_checkpoint), filename=fname)
                sd = torch.load(p, map_location="cpu")
            except Exception as e:
                print(f"  [Skip] {fname}: {e}")
        if sd is not None:
            current = getattr(state, sum_attr)
            for k, v in sd.items():
                if k in current:
                    current[k] = current[k] + v.cpu()
                else:
                    current[k] = v.cpu().clone()
            print(f"  [OK] {fname} accumulated ({len(sd)} keys)")

    del pretrained_sd, task_sd, tau_t
    torch.cuda.empty_cache()

    print(f"[Step {t}] Merge step complete. Total tasks merged: {state.task_count}")
    return state


def assemble_model(
    state: ContinualMergeState,
    base_checkpoint: Union[str, Path],
    device: str = "cpu",
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForVision2Seq:
    print(f"[Assemble] Loading base model for assembly...")
    base_model = load_model(
        base_checkpoint,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device=device,
    )
    merged_sd = {k: v.clone() for k, v in base_model.state_dict().items()}

    T = state.task_count
    print(f"[Assemble] Applying averaged task vectors (T={T})...")
    for k, tau_sum in state.tau_sum.items():
        if k not in merged_sd:
            continue
        avg_delta = tau_sum / T
        merged_sd[k] = (merged_sd[k].to("cpu", torch_dtype) + avg_delta.to("cpu", torch_dtype))

    base_model.load_state_dict(merged_sd, strict=False)
    del merged_sd
    return base_model


def save_step_checkpoint(
    state:           ContinualMergeState,
    base_checkpoint: Union[str, Path],
    save_dir:        Path,
    trust_remote_code: bool = True,
    torch_dtype:     torch.dtype = torch.bfloat16,
    save_processor:  bool = True,
    last_task_checkpoint: Optional[Union[str, Path]] = None,
):

    t = state.task_count
    step_dir = save_dir / f"step_{t}"
    step_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Save] Assembling merged model for step_{t}...")
    start = time.time()

    merged_model = assemble_model(
        state,
        base_checkpoint=base_checkpoint,
        device="cpu",
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    merged_model.save_pretrained(step_dir)
    print(f"[Save] Model saved  ({time.time()-start:.1f}s) → {step_dir}")

    if save_processor:
        proc_source = last_task_checkpoint or base_checkpoint
        try:
            processor = load_processor(proc_source, trust_remote_code=trust_remote_code)
            processor.save_pretrained(step_dir)
            print(f"[Save] Processor saved → {step_dir}")
        except Exception as e:
            print(f"[Save] Processor save skipped: {e}")

    T = state.task_count
    for fname, sum_attr in [("proprio_projector--checkpoint.pt", "proprio_sum"),
                             ("action_head--checkpoint.pt", "action_head_sum")]:
        current = getattr(state, sum_attr)
        if current:
            avg_sd = {k: v / T for k, v in current.items()}
            torch.save(avg_sd, step_dir / fname)
            print(f"[Save] Saved averaged {fname} (T={T}) → {step_dir}")

    meta = {
        "step":       t,
        "task_names": state.task_names,
        "merged_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_checkpoint": str(base_checkpoint),
    }
    with open(step_dir / "merge_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    state.save(save_dir / "continual_state.pt")

    del merged_model
    torch.cuda.empty_cache()
    print(f"[Save] step_{t} checkpoint complete → {step_dir}")


def continuous_merge(
    task_order:       List[str],
    task_checkpoints: Dict[str, Union[str, Path]],
    base_checkpoint:  Union[str, Path],
    save_dir:         Union[str, Path] = "outputs/continual",
    resume:           bool = False,
    trust_remote_code: bool = True,
    torch_dtype:      torch.dtype = torch.bfloat16,
    device:           str = "cpu",
    note:             Optional[str] = None,
) -> ContinualMergeState:

    save_path = Path(save_dir)
    order_tag = "_".join(task_order)
    folder_name = f"order_{order_tag}" + (f"_{note}" if note else "")
    save_path = save_path / folder_name
    save_path.mkdir(parents=True, exist_ok=True)

    config_log = {
        "task_order":       task_order,
        "task_checkpoints": {k: str(v) for k, v in task_checkpoints.items()},
        "base_checkpoint":  str(base_checkpoint),
        "merge_method":     "average",
    }
    with open(save_path / "run_config.json", "w") as f:
        json.dump(config_log, f, indent=2, ensure_ascii=False)
    print(f"\n[Config] Run config saved → {save_path / 'run_config.json'}")

    state_path = save_path / "continual_state.pt"
    if resume and state_path.exists():
        state = ContinualMergeState.load(state_path)
        start_step = state.task_count
        print(f"[Resume] Resuming from step {start_step}")
    else:
        state = None
        start_step = 0

    for step_idx in range(start_step, len(task_order)):
        task_name = task_order[step_idx]

        if task_name not in task_checkpoints:
            raise ValueError(
                f"task '{task_name}' not found in task_checkpoints. "
                f"Available: {list(task_checkpoints.keys())}"
            )

        state = merge_step(
            task_name=task_name,
            task_checkpoint=task_checkpoints[task_name],
            base_checkpoint=base_checkpoint,
            state=state,
            device=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )

        save_step_checkpoint(
            state=state,
            base_checkpoint=base_checkpoint,
            save_dir=save_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            save_processor=True,
            last_task_checkpoint=task_checkpoints[task_name],
        )

    print(f"\n{'='*60}")
    print(f" Continuous merge complete. {state.task_count} tasks merged.")
    print(f" Checkpoints saved under: {save_path}")
    print(f"{'='*60}")
    return state

if __name__ == "__main__":

    BASE_CHECKPOINT = os.environ.get("BASE_CHECKPOINT", "openvla/openvla-7b")

    TASK_CHECKPOINTS = {
        "spatial": os.environ.get("CKPT_SPATIAL", "VLA-Adapter/LIBERO-Spatial-Pro"),
        "object":  os.environ.get("CKPT_OBJECT",  "VLA-Adapter/LIBERO-Object-Pro"),
        "goal":    os.environ.get("CKPT_GOAL",    "VLA-Adapter/LIBERO-Goal-Pro"),
        "10":      os.environ.get("CKPT_10",      "VLA-Adapter/LIBERO-Long-Pro"),
    }

    TASK_ORDER = os.environ.get("TASK_ORDER", "spatial,object,goal,10").split(",")

    SAVE_DIR = os.environ.get("SAVE_DIR", "./continual_merged_models/VLAAdapter-pro")

    DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE      = torch.bfloat16
    TRUST_REMOTE     = True    
    RESUME           = False    
    NOTE             = None    

    continuous_merge(
        task_order=TASK_ORDER,
        task_checkpoints=TASK_CHECKPOINTS,
        base_checkpoint=BASE_CHECKPOINT,
        save_dir=SAVE_DIR,
        resume=RESUME,
        trust_remote_code=TRUST_REMOTE,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
        note=NOTE,
    )