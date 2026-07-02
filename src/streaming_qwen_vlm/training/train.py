"""Joint VLA training: FAST autoregressive loss (backbone) + flow matching (expert).

    PYTHONPATH=src python -m streaming_qwen_vlm.training.train --out-dir checkpoints/run1
    PYTHONPATH=src python -m streaming_qwen_vlm.training.train \
        --overfit-episode 0 --steps 2000 --out-dir checkpoints/overfit    # M2 gate

Single H200. AdamW fp32 params + bf16 autocast, per-layer gradient checkpointing (ViT + LLM),
param groups: backbone @ lr_backbone, FAST side-car + expert @ lr_new. EMA on the expert only.
"""

from __future__ import annotations

import functools
import json
import math
import os
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..backbone import load_backbone
from ..config import VLMConfig
from ..expert import ExpertConfig
from ..fast_tokens import FastActionTokenizer
from ..normalize import compute_norm_stats, load_stats, save_stats, split_episodes
from ..vla import QwenVLA
from . import checkpoints, evaluate
from .config import TrainConfig, parse_args
from .dataset import TrossenActDataset, collate


class _NullLogger:
    def log(self, *a, **k):
        pass

    def finish(self):
        pass


def _make_logger(tc: TrainConfig):
    try:
        import wandb

        return wandb.init(project=tc.wandb_project, name=tc.run_name,
                          config={f: getattr(tc, f) for f in vars(tc)}, dir=tc.out_dir)
    except Exception as exc:  # wandb missing / offline cluster
        print(f"[train] wandb unavailable ({exc}); falling back to metrics.jsonl")
        return _NullLogger()


def _lr_lambda(step: int, warmup: int, total: int, final_frac: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def main() -> None:
    tc = parse_args()
    os.makedirs(tc.out_dir, exist_ok=True)
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    vlm_cfg = VLMConfig() if tc.instruction is None else VLMConfig(instruction=tc.instruction)
    assert vlm_cfg.num_state_tokens == 0, "Stage 2 uses state-as-text; num_state_tokens must be 0"

    # Norm stats (train split only).
    if os.path.exists(tc.norm_stats):
        stats = load_stats(tc.norm_stats)
    else:
        train_eps, val_eps = split_episodes(tc.num_episodes, tc.holdout_every)
        stats = compute_norm_stats(tc.data_root, train_eps)
        stats["train_episodes"], stats["val_episodes"] = train_eps, val_eps
        save_stats(tc.norm_stats, {"action": stats["action"], "state": stats["state"]},
                   train_eps, val_eps)
    if tc.overfit_episode is not None:
        train_eps = val_eps = [tc.overfit_episode]
    else:
        train_eps, val_eps = stats["train_episodes"], stats["val_episodes"]

    # FAST tokenizer (also validates the download; pin --fast-revision after the first run).
    fast = FastActionTokenizer(tc.fast_repo, tc.fast_revision, horizon=tc.horizon,
                               action_dim=vlm_cfg.state_dim)
    print(f"[train] FAST vocab_size={fast.vocab_size}")

    # Model: bf16+FA2 load, then fp32 master params for training.
    backbone, processor = load_backbone(vlm_cfg)
    backbone = backbone.float()
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    expert_cfg = ExpertConfig(width=tc.expert_width, mlp_dim=tc.expert_mlp_dim, horizon=tc.horizon)
    vla = QwenVLA(backbone, processor, vlm_cfg, expert_cfg, fast.vocab_size).to(device)
    vla.train()
    n_bb = sum(p.numel() for p in vla.backbone.parameters())
    n_new = sum(p.numel() for p in vla.expert.parameters()) + vla.fast_embed.weight.numel() \
        + vla.fast_head.weight.numel()
    print(f"[train] backbone {n_bb/1e9:.2f}B params, expert+side-car {n_new/1e6:.0f}M params, "
          f"S_prefix={vla.S_prefix}")

    ds_kwargs = dict(
        root=tc.data_root, cfg=vlm_cfg, processor=processor, state_lut=vla.template.state_lut,
        action_stats=stats["action"], state_stats=stats["state"],
        max_fast_tokens=tc.max_fast_tokens, horizon=tc.horizon,
        fast_repo=tc.fast_repo, fast_revision=tc.fast_revision,
    )
    train_ds = TrossenActDataset(episodes=train_eps, **ds_kwargs)
    val_ds = TrossenActDataset(episodes=val_eps, tick_aligned=True, **ds_kwargs)
    print(f"[train] {len(train_ds)} train samples ({len(train_eps)} eps), "
          f"{len(val_ds)} val ticks ({len(val_eps)} eps)")
    collate_fn = functools.partial(collate, cfg=vlm_cfg)
    train_loader = DataLoader(train_ds, batch_size=tc.micro_batch, shuffle=True, drop_last=True,
                              num_workers=tc.num_workers, collate_fn=collate_fn, pin_memory=True,
                              persistent_workers=tc.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=tc.micro_batch, shuffle=False,
                            num_workers=min(4, tc.num_workers), collate_fn=collate_fn,
                            pin_memory=True)

    optimizer = torch.optim.AdamW(vla.param_groups(tc.lr_backbone, tc.lr_new),
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=functools.partial(_lr_lambda, warmup=tc.warmup_steps, total=tc.steps,
                                    final_frac=tc.lr_final_frac),
    )
    all_params = [p for g in optimizer.param_groups for p in g["params"]]
    ema_state = {n: p.detach().float().clone() for n, p in vla.expert.named_parameters()}

    start_step = 0
    if tc.resume:
        start_step, ema_state = checkpoints.load_resume_state(tc.out_dir, vla, optimizer,
                                                              scheduler, device)
        print(f"[train] resumed from step {start_step}")

    logger = _make_logger(tc)
    metrics_path = os.path.join(tc.out_dir, "metrics.jsonl")
    data_iter = iter(train_loader)

    def next_batch() -> Dict[str, torch.Tensor]:
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    skipped = 0
    t_last = time.time()
    for step in range(start_step, tc.steps):
        agg = {"loss": 0.0, "loss_ar": 0.0, "loss_flow": 0.0, "fast_acc": 0.0}
        for _ in range(tc.grad_accum):
            batch = next_batch()
            # cache_enabled=False is REQUIRED: the no_grad KV export in llm_forward would otherwise
            # populate the autocast weight cache with grad-disconnected bf16 casts, breaking the
            # checkpointed layer forwards (CheckpointError in backward / missing k,v,norm grads).
            # Costs ~nothing: each weight is cast once per micro-forward either way.
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                out = vla.forward_train(batch)
            (out["loss"] / tc.grad_accum).backward()
            for k in agg:
                agg[k] += float(out[k]) / tc.grad_accum

        total_norm = torch.nn.utils.clip_grad_norm_(all_params, tc.grad_clip)
        if torch.isfinite(total_norm):
            optimizer.step()
            with torch.no_grad():
                for n, p in vla.expert.named_parameters():
                    ema_state[n].mul_(tc.ema_decay).add_(p.float(), alpha=1.0 - tc.ema_decay)
        else:
            skipped += 1
            print(f"[train] step {step}: non-finite grad norm, skipping optimizer step ({skipped})")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % tc.log_every == 0:
            dt = time.time() - t_last
            t_last = time.time()
            eff_batch = tc.micro_batch * tc.grad_accum
            metrics = {
                **agg,
                "grad_norm": float(total_norm),
                "lr_backbone": scheduler.get_last_lr()[0],
                "lr_new": scheduler.get_last_lr()[1],
                "samples_per_s": eff_batch * tc.log_every / dt,
                "skipped_steps": skipped,
                "cuda_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
            logger.log(metrics, step=step + 1)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step + 1, **metrics}) + "\n")
            print(f"[{step + 1}/{tc.steps}] loss={agg['loss']:.4f} "
                  f"(ar={agg['loss_ar']:.4f} flow={agg['loss_flow']:.4f}) "
                  f"acc={agg['fast_acc']:.3f} {metrics['samples_per_s']:.2f} samp/s "
                  f"{metrics['cuda_gb']:.1f} GB")

        if (step + 1) % tc.eval_every == 0 or step + 1 == tc.steps:
            val = evaluate.run(vla, val_loader, device, ema_state, max_batches=tc.eval_batches,
                               num_denoise_steps=vlm_cfg.num_denoise_steps)
            logger.log(val, step=step + 1)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step + 1, **val}) + "\n")
            print(f"[eval @ {step + 1}] " + "  ".join(
                f"{k}={v:.4f}" for k, v in val.items() if not k.startswith("val_openloop_mse_joint")))

        if (step + 1) % tc.save_every == 0 or step + 1 == tc.steps:
            path = checkpoints.save_step_checkpoint(tc.out_dir, step + 1, vla, processor,
                                                    ema_state, tc, tc.norm_stats, tc.keep_last)
            checkpoints.save_resume_state(tc.out_dir, step + 1, vla, optimizer, scheduler,
                                          ema_state)
            print(f"[train] saved {path}")

    logger.finish() if hasattr(logger, "finish") else None
    print("[train] done")


if __name__ == "__main__":
    main()
