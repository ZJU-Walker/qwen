"""Held-out evaluation: AR token accuracy, val losses, 10-step denoised open-loop MSE."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional

import torch

from ..flow import sample_actions


@contextmanager
def use_ema(expert, ema_state: Optional[Dict[str, torch.Tensor]]):
    """Temporarily swap expert parameters with their EMA copies."""
    if ema_state is None:
        yield
        return
    backup = {n: p.detach().clone() for n, p in expert.named_parameters()}
    with torch.no_grad():
        for n, p in expert.named_parameters():
            p.copy_(ema_state[n].to(p.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in expert.named_parameters():
                p.copy_(backup[n])


@torch.no_grad()
def run(vla, val_loader, device, ema_state: Optional[Dict[str, torch.Tensor]] = None,
        max_batches: int = 8, num_denoise_steps: int = 10) -> Dict[str, float]:
    was_training = vla.training
    vla.eval()
    gen = torch.Generator(device=device).manual_seed(0)  # fixed noise -> low-variance val curves

    totals = {"val_loss_ar": 0.0, "val_loss_flow": 0.0, "val_fast_acc": 0.0, "val_openloop_mse": 0.0}
    per_joint = torch.zeros(vla.expert_cfg.action_dim, device=device)
    n = 0
    with use_ema(vla.expert, ema_state):
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=vla.compute_dtype):
                out = vla.forward_train(batch, generator=gen)
                chunk = sample_actions(
                    vla.expert, out["prefix_kv"], vla.suffix_pos_emb(batch["state_ids"].shape[0]),
                    vla.expert_cfg.horizon, vla.expert_cfg.action_dim, num_denoise_steps, gen,
                )
            err2 = (chunk - batch["actions_norm"].float()).pow(2)
            totals["val_loss_ar"] += float(out["loss_ar"])
            totals["val_loss_flow"] += float(out["loss_flow"])
            totals["val_fast_acc"] += float(out["fast_acc"])
            totals["val_openloop_mse"] += float(err2.mean())
            per_joint += err2.mean(dim=(0, 1))
            n += 1

    if was_training:
        vla.train()
    if n == 0:
        return {}
    metrics = {k: v / n for k, v in totals.items()}
    for j, v in enumerate((per_joint / n).tolist()):
        metrics[f"val_openloop_mse_joint{j}"] = v
    return metrics
