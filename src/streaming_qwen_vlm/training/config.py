"""Training configuration (json-serializable dataclass; every field is an argparse flag)."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    # Data
    data_root: str = "/iris/projects/humanoid/trossen_data/0528_merge_block_mem"
    norm_stats: str = "checkpoints/norm_stats.json"
    num_episodes: int = 61
    holdout_every: int = 10
    horizon: int = 30
    instruction: Optional[str] = None       # None -> VLMConfig default
    # Vision config (per-run; saved in checkpoint meta so serving adapts automatically):
    resolution: int = 336                   # square input, multiple of 28 (224 -> 64 tok/pair)
    fps: int = 3                            # sampled frame rate; must divide the 30 Hz data rate
    num_pairs: int = 15                     # window length in 2-frame pairs
    num_workers: int = 12

    # FAST
    fast_repo: str = "physical-intelligence/fast"
    fast_revision: Optional[str] = None     # pin after the first download
    max_fast_tokens: int = 192              # measured train-split max 147 (+30% headroom)

    # Expert
    expert_width: int = 1024                # fallback 768 (~500M) if M3 overfits
    expert_mlp_dim: int = 4096

    # Optimization (claude_plan §3)
    steps: int = 30_000
    warmup_steps: int = 1_000
    micro_batch: int = 8
    grad_accum: int = 4                     # effective batch 32
    lr_backbone: float = 2e-5
    lr_new: float = 1e-4                    # FAST side-car + expert
    lr_final_frac: float = 0.1
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999                # expert only (full-model EMA costs ~18 GB)
    seed: int = 0

    # Speed knobs (opt-in; defaults = original behavior)
    ckpt_stride: int = 1        # recompute every N-th LLM layer only; 2 = ~+10% speed, +~29 GB
    compile: bool = False       # regional torch.compile of the 36 decoder layers (~1-3 min warmup)

    # Cadence / output
    log_every: int = 20
    eval_every: int = 1_000
    eval_batches: int = 8
    save_every: int = 500       # deployable step_XXXXXX snapshot (11 GB) cadence
    keep_every: int = 2_000     # snapshots at these multiples are kept forever (milestones)
    keep_last: int = 3          # newest non-milestone snapshots kept
    resume_every: int = 2_000   # fp32 resume bundle (55 GB) cadence — heavier, so rarer
    out_dir: str = "checkpoints/run1"
    tmp_dir: Optional[str] = None             # local scratch for multiprocessing temp files
    wandb_project: str = "qwen-vla"
    run_name: Optional[str] = None
    resume: bool = False                    # resume from <out_dir>/latest

    # Gate mode (M2): restrict to one episode, no val split
    overfit_episode: Optional[int] = None


def parse_args(argv=None) -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train the Qwen VLA (FAST AR + flow expert).")
    for f in dataclasses.fields(TrainConfig):
        name = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(name, action="store_true" if not f.default else "store_false")
        elif f.type in ("Optional[int]",) or f.name == "overfit_episode":
            ap.add_argument(name, type=int, default=f.default)
        elif f.type in ("Optional[str]",) or f.default is None:
            ap.add_argument(name, type=str, default=f.default)
        else:
            ap.add_argument(name, type=type(f.default), default=f.default)
    ns = ap.parse_args(argv)
    return TrainConfig(**vars(ns))
