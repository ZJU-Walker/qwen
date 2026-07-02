"""The action expert: a pi0.5-faithful parallel transformer with joint per-layer attention.

At every layer L, the expert's 30 suffix (noisy-action) queries attend over
``[prefix K_L,V_L (Qwen, detached) ++ suffix K_L,V_L (expert)]`` with ONE softmax over the
concatenated sequence (claude_plan D1). Constraints ported from openpi (gemma.py asserts the
same): heads / kv_heads / head_dim MUST equal the backbone's (16 / 2 / 128) so the K/V layouts
concatenate; only width and MLP shrink.

Positions: the suffix continues right after the prefix max mrope position on ALL THREE axes
("text mode"). This range deliberately OVERLAPS the FAST-token range used during training — the
two suffixes never attend to each other, and FAST tokens do not exist at inference. Suffix q AND
k are roped with Qwen's own rotary tables; the exported prefix K arrives already roped and is
NEVER re-roped here.

adaRMS conditioning on the flow time tau follows openpi's adarms variant: every norm (including
the final one) gets zero-init (scale, shift, gate) from a Linear(width -> 3*width); at init each
block is an exact identity, so the expert starts near-identity and ignores the prefix (adaLN-zero).
The MLP is SiLU-gated per the locked plan (openpi's gemma uses GELU gating — immaterial, recorded).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from .flow import posemb_sincos


@dataclass
class ExpertConfig:
    width: int = 1024
    depth: int = 36                    # must equal backbone num_hidden_layers
    mlp_dim: int = 4096
    num_heads: int = 16                # must equal backbone num_attention_heads
    num_kv_heads: int = 2              # must equal backbone num_key_value_heads
    head_dim: int = 128                # must equal backbone hidden_size / num_attention_heads
    rms_eps: float = 1e-6
    horizon: int = 30
    action_dim: int = 14
    min_period: float = 4e-3
    max_period: float = 4.0
    mrope_section: Tuple[int, int, int] = (16, 24, 24)  # must equal backbone rope_scaling

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError("sum(mrope_section) must equal head_dim / 2")


class AdaRMSNorm(nn.Module):
    """RMSNorm whose scale/shift/gate come from the tau conditioning vector (zero-init).

    forward(x [B,T,W], cond [B,W]) -> (normed [B,T,W], gate [B,1,W]). Variance in fp32. With the
    zero-init modulation the output is plain RMS-normed x and gate == 0 (identity blocks at init).
    """

    def __init__(self, width: int, cond_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.mod = nn.Linear(cond_dim, 3 * width)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        scale, shift, gate = self.mod(cond).float().unsqueeze(1).chunk(3, dim=-1)  # each [B,1,W]
        out = normed * (1.0 + scale) + shift
        return out.to(x.dtype), gate.to(x.dtype)


class ExpertBlock(nn.Module):
    def __init__(self, cfg: ExpertConfig) -> None:
        super().__init__()
        self.cfg = cfg
        W, H, KV, hd = cfg.width, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(W, H * hd, bias=False)
        self.k_proj = nn.Linear(W, KV * hd, bias=False)
        self.v_proj = nn.Linear(W, KV * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, W, bias=False)
        self.gate_proj = nn.Linear(W, cfg.mlp_dim, bias=False)
        self.up_proj = nn.Linear(W, cfg.mlp_dim, bias=False)
        self.down_proj = nn.Linear(cfg.mlp_dim, W, bias=False)
        self.pre_attn_norm = AdaRMSNorm(W, W, cfg.rms_eps)
        self.pre_mlp_norm = AdaRMSNorm(W, W, cfg.rms_eps)

    def forward(
        self,
        x: torch.Tensor,                                   # [B, T, W] suffix stream
        k_pre: torch.Tensor,                               # [B, KV, S_p, hd] Qwen K (roped, detached)
        v_pre: torch.Tensor,                               # [B, KV, S_p, hd]
        pos_emb: Tuple[torch.Tensor, torch.Tensor],        # (cos, sin) [3, B, T, hd] at suffix positions
        cond: torch.Tensor,                                # [B, W] tau conditioning
    ) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = x.shape
        H, KV, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim

        h, gate = self.pre_attn_norm(x, cond)
        q = self.q_proj(h).view(B, T, H, hd).transpose(1, 2)
        ks = self.k_proj(h).view(B, T, KV, hd).transpose(1, 2)
        vs = self.v_proj(h).view(B, T, KV, hd).transpose(1, 2)

        cos, sin = pos_emb
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        # Suffix q AND suffix k get Qwen rope at the suffix positions; prefix K is already roped.
        q, ks = apply_multimodal_rotary_pos_emb(q, ks, cos, sin, list(cfg.mrope_section))

        K = torch.cat([k_pre.to(q.dtype), ks], dim=2)      # [B, KV, S_p + T, hd]
        V = torch.cat([v_pre.to(q.dtype), vs], dim=2)
        K = repeat_kv(K, H // KV)
        V = repeat_kv(V, H // KV)

        # One softmax over [prefix ++ suffix]; suffix is bidirectional within itself and sees the
        # full prefix, so with a constant S_prefix no mask is needed at all.
        a = F.scaled_dot_product_attention(q, K, V, attn_mask=None, is_causal=False)
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, T, H * hd)) * gate

        h, gate = self.pre_mlp_norm(x, cond)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)) * gate
        return x


class ActionExpert(nn.Module):
    def __init__(self, cfg: ExpertConfig) -> None:
        super().__init__()
        self.cfg = cfg
        W = cfg.width
        self.action_in_proj = nn.Linear(cfg.action_dim, W)
        self.time_mlp_in = nn.Linear(W, W)
        self.time_mlp_out = nn.Linear(W, W)
        self.blocks = nn.ModuleList(ExpertBlock(cfg) for _ in range(cfg.depth))
        self.final_norm = AdaRMSNorm(W, W, cfg.rms_eps)
        self.action_out_proj = nn.Linear(W, cfg.action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        # lecun-normal everywhere, then zero every adaRMS modulation so blocks start as identities.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=math.sqrt(1.0 / m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, AdaRMSNorm):
                nn.init.zeros_(m.mod.weight)
                nn.init.zeros_(m.mod.bias)

    def time_cond(self, tau: torch.Tensor) -> torch.Tensor:
        """tau fp32 [B] -> [B, width]: posemb_sincos -> Linear -> swish -> Linear -> swish."""
        emb = posemb_sincos(tau, self.cfg.width, self.cfg.min_period, self.cfg.max_period)
        emb = emb.to(self.time_mlp_in.weight.dtype)
        return F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(emb))))

    def forward(
        self,
        x_tau: torch.Tensor,                               # [B, horizon, action_dim] noisy actions
        tau: torch.Tensor,                                 # [B] flow time
        prefix_kv: List[Tuple[torch.Tensor, torch.Tensor]],  # per-layer detached Qwen (K, V)
        suffix_pos_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.cfg
        if len(prefix_kv) != cfg.depth:
            raise ValueError(f"prefix_kv has {len(prefix_kv)} layers, expert expects {cfg.depth}")
        x = self.action_in_proj(x_tau.to(self.action_in_proj.weight.dtype))
        cond = self.time_cond(tau)
        for block, (k_pre, v_pre) in zip(self.blocks, prefix_kv):
            x = block(x, k_pre, v_pre, suffix_pos_emb, cond)
        y, _ = self.final_norm(x, cond)
        return self.action_out_proj(y)
