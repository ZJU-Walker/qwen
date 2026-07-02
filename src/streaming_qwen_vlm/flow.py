"""Flow matching for the action expert — exact openpi pi0.py conventions.

    tau ~ Beta(1.5, 1) * 0.999 + 0.001      # tau = 1 is PURE NOISE (code convention, not paper)
    x_tau = tau * eps + (1 - tau) * a,  eps ~ N(0, I)
    target u = eps - a
    L_flow = MSE(v_theta(x_tau, tau, sg[prefix K/V]), u)      (computed in fp32)
    inference: 10 Euler steps, x <- x + dt * v, dt = -1/10, tau: 1 -> 0
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


def posemb_sincos(pos: torch.Tensor, dim: int, min_period: float, max_period: float) -> torch.Tensor:
    """Exact port of openpi's posemb_sincos: pos [B] -> [B, dim] fp32 (sin half, cos half)."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    pos = pos.to(torch.float32)
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=pos.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    angle = pos[:, None] * (1.0 / period)[None, :] * (2.0 * math.pi)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


def sample_tau(batch: int, device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """tau ~ Beta(1.5, 1) * 0.999 + 0.001 via inverse-CDF (Beta(a,1) CDF is x^a) -> fp32 [B]."""
    u = torch.rand(batch, device=device, generator=generator)
    return u.pow(1.0 / 1.5) * 0.999 + 0.001


def flow_loss(
    expert,
    prefix_kv,
    suffix_pos_emb: Tuple[torch.Tensor, torch.Tensor],
    actions: torch.Tensor,  # [B, horizon, action_dim] normalized, fp32
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B = actions.shape[0]
    device = actions.device
    eps = torch.randn(actions.shape, device=device, generator=generator, dtype=torch.float32)
    tau = sample_tau(B, device, generator)

    t = tau[:, None, None]
    x_tau = t * eps + (1.0 - t) * actions
    u = eps - actions

    v = expert(x_tau, tau, prefix_kv, suffix_pos_emb)
    loss = torch.nn.functional.mse_loss(v.float(), u.float())
    return loss, {"flow_mse": float(loss.detach())}


@torch.no_grad()
def sample_actions(
    expert,
    prefix_kv,
    suffix_pos_emb: Tuple[torch.Tensor, torch.Tensor],
    horizon: int,
    action_dim: int,
    num_steps: int = 10,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """10-step Euler integration from pure noise (tau=1) to the action chunk (tau=0). -> [B, H, A] fp32."""
    B = prefix_kv[0][0].shape[0]
    device = prefix_kv[0][0].device
    dt = -1.0 / num_steps

    x = torch.randn(B, horizon, action_dim, device=device, generator=generator, dtype=torch.float32)
    tau = 1.0
    while tau >= -dt / 2.0:
        v = expert(x, torch.full((B,), tau, device=device, dtype=torch.float32), prefix_kv, suffix_pos_emb)
        x = x + dt * v.float()
        tau += dt
    return x
