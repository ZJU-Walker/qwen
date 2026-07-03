"""Checkpoint save/resume.

Two artifact kinds:
- ``<out_dir>/step_XXXXXX/`` — deployable weights-only snapshots (bf16): backbone in stock
  from_pretrained layout (tied lm_head dropped; re-tied on load), expert + FAST side-car, expert
  EMA, meta.json (configs, S_prefix, fast vocab), norm_stats.json copy. Pruned to ``keep_last``.
- ``<out_dir>/latest/`` — full-fidelity resume bundle (fp32 model state dict, optimizer,
  scheduler, EMA, step counter), overwritten each save.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file


def _init_field_dict(dc) -> dict:
    """dataclass -> {init-field: value} (derived init=False fields excluded so **kwargs round-trips)."""
    return {f.name: getattr(dc, f.name) for f in dataclasses.fields(dc) if f.init}


def _heads_state_dict(vla, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    sd = {}
    for prefix, module in (("expert.", vla.expert), ("fast_embed.", vla.fast_embed),
                           ("fast_head.", vla.fast_head)):
        for k, v in module.state_dict().items():
            sd[prefix + k] = v.detach().to(dtype).cpu()
    return sd


def save_step_checkpoint(out_dir: str, step: int, vla, processor, ema_state: Dict[str, torch.Tensor],
                         train_cfg, norm_stats_path: str, keep_last: int = 3,
                         keep_every: int = 0) -> str:
    path = os.path.join(out_dir, f"step_{step:06d}")
    bdir = os.path.join(path, "backbone")
    os.makedirs(bdir, exist_ok=True)

    # Backbone, stock-shaped, bf16. lm_head.weight is tied to the embedding: drop it so
    # safetensors sees no shared tensors; from_pretrained re-ties via tie_word_embeddings.
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in vla.backbone.state_dict().items()
          if k != "lm_head.weight"}
    vla.backbone.save_pretrained(bdir, state_dict=sd, safe_serialization=True)
    processor.save_pretrained(bdir)

    save_file(_heads_state_dict(vla, torch.bfloat16), os.path.join(path, "vla_heads.safetensors"))
    save_file({k: v.to(torch.bfloat16).cpu() for k, v in ema_state.items()},
              os.path.join(path, "expert_ema.safetensors"))

    meta = {
        "step": step,
        "S_prefix": vla.S_prefix,
        "fast_vocab_size": vla.fast_vocab_size,
        "vlm_config": _init_field_dict(vla.cfg),
        "expert_config": _init_field_dict(vla.expert_cfg),
        "train_config": _init_field_dict(train_cfg),
    }
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    if os.path.exists(norm_stats_path):
        shutil.copy(norm_stats_path, os.path.join(path, "norm_stats.json"))

    _prune(out_dir, keep_last, keep_every)
    return path


def _prune(out_dir: str, keep_last: int, keep_every: int = 0) -> None:
    """Delete step_XXXXXX dirs that are neither keep_every milestones nor the newest keep_last."""
    steps = sorted(d for d in os.listdir(out_dir) if d.startswith("step_"))
    recent = set(steps[-keep_last:]) if keep_last > 0 else set()
    for d in steps:
        if d in recent:
            continue
        if keep_every > 0 and int(d[len("step_"):]) % keep_every == 0:
            continue  # permanent milestone
        shutil.rmtree(os.path.join(out_dir, d), ignore_errors=True)


def save_resume_state(out_dir: str, step: int, vla, optimizer, scheduler,
                      ema_state: Dict[str, torch.Tensor]) -> None:
    path = os.path.join(out_dir, "latest")
    tmp = path + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    torch.save(vla.state_dict(), os.path.join(tmp, "model_fp32.pt"))
    torch.save(optimizer.state_dict(), os.path.join(tmp, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(tmp, "scheduler.pt"))
    torch.save(ema_state, os.path.join(tmp, "ema.pt"))
    with open(os.path.join(tmp, "step.json"), "w") as f:
        json.dump({"step": step}, f)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.rename(tmp, path)


def load_resume_state(out_dir: str, vla, optimizer, scheduler,
                      device) -> Tuple[int, Optional[Dict[str, torch.Tensor]]]:
    path = os.path.join(out_dir, "latest")
    vla.load_state_dict(torch.load(os.path.join(path, "model_fp32.pt"), map_location=device))
    optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt"), map_location=device))
    scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt")))
    ema = torch.load(os.path.join(path, "ema.pt"), map_location=device)
    with open(os.path.join(path, "step.json")) as f:
        step = json.load(f)["step"]
    return step, ema
