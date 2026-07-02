"""T3: expert joint attention vs a naive dense reference — CPU, tiny dims.

Verifies with an independent implementation (explicit softmax, hand-rolled rope, GQA loops) that
an ExpertBlock computes: one softmax over [prefix K/V ++ suffix K/V], suffix bidirectional within
itself and attending the full prefix, suffix q AND k roped at suffix positions while the prefix K
is used as-given (never re-roped), residuals gated by the adaRMS gates.
"""

import math

import torch
import torch.nn as nn

from streaming_qwen_vlm.expert import AdaRMSNorm, ExpertBlock, ExpertConfig

torch.manual_seed(0)

CFG = ExpertConfig(
    width=32, depth=2, mlp_dim=64, num_heads=4, num_kv_heads=2, head_dim=8,
    horizon=5, action_dim=3, mrope_section=(1, 1, 2),  # sums to head_dim//2 = 4
)
B, T, S_P = 2, 5, 7


def _rope_tables(positions: torch.Tensor, head_dim: int):
    """Standard rope tables, expanded to the [3, B, T, head_dim] mrope layout (same on all axes)."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]          # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                          # [T, head_dim]
    cos = emb.cos()[None, None].expand(3, B, T, head_dim).contiguous()
    sin = emb.sin()[None, None].expand(3, B, T, head_dim).contiguous()
    return cos, sin


def _rope_apply_ref(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Independent rope: x [B, H, T, hd]. With identical mrope axes this equals Qwen's mrope."""
    hd = x.shape[-1]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat([freqs, freqs], -1).cos()[None, None]
    sin = torch.cat([freqs, freqs], -1).sin()[None, None]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


def _rms_ref(x: torch.Tensor, mod: nn.Linear, cond: torch.Tensor):
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    scale, shift, gate = mod(cond).unsqueeze(1).chunk(3, -1)
    return normed * (1 + scale) + shift, gate


def _block_ref(block: ExpertBlock, x, k_pre, v_pre, positions, cond):
    """Dense reference forward: explicit GQA expansion + softmax over the concatenated sequence."""
    cfg = block.cfg
    H, KV, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim

    h, gate = _rms_ref(x, block.pre_attn_norm.mod, cond)
    q = block.q_proj(h).view(B, T, H, hd).permute(0, 2, 1, 3)
    ks = block.k_proj(h).view(B, T, KV, hd).permute(0, 2, 1, 3)
    vs = block.v_proj(h).view(B, T, KV, hd).permute(0, 2, 1, 3)
    q = _rope_apply_ref(q, positions)
    ks = _rope_apply_ref(ks, positions)      # suffix k roped; prefix K used as-given

    K = torch.cat([k_pre, ks], dim=2)        # [B, KV, S_P+T, hd]
    V = torch.cat([v_pre, vs], dim=2)
    out = torch.zeros(B, H, T, hd)
    for b in range(B):
        for hh in range(H):
            kv_head = hh // (H // KV)        # GQA: each kv head serves H/KV query heads
            scores = q[b, hh] @ K[b, kv_head].T / math.sqrt(hd)   # [T, S_P+T] — NO mask anywhere
            out[b, hh] = torch.softmax(scores, dim=-1) @ V[b, kv_head]
    x = x + block.o_proj(out.permute(0, 2, 1, 3).reshape(B, T, H * hd)) * gate

    h, gate = _rms_ref(x, block.pre_mlp_norm.mod, cond)
    x = x + block.down_proj(torch.nn.functional.silu(block.gate_proj(h)) * block.up_proj(h)) * gate
    return x


def test_expert_block_matches_dense_reference():
    block = ExpertBlock(CFG)
    # Randomize the adaRMS modulations (zero-init would hide attention/MLP entirely).
    for m in block.modules():
        if isinstance(m, AdaRMSNorm):
            nn.init.normal_(m.mod.weight, std=0.2)
            nn.init.normal_(m.mod.bias, std=0.2)

    x = torch.randn(B, T, CFG.width)
    cond = torch.randn(B, CFG.width)
    k_pre = torch.randn(B, CFG.num_kv_heads, S_P, CFG.head_dim)
    v_pre = torch.randn(B, CFG.num_kv_heads, S_P, CFG.head_dim)
    positions = torch.arange(100, 100 + T)   # suffix continues after some prefix max
    pos_emb = _rope_tables(positions, CFG.head_dim)

    with torch.no_grad():
        got = block(x, k_pre, v_pre, pos_emb, cond)
        ref = _block_ref(block, x, k_pre, v_pre, positions, cond)

    diff = (got - ref).abs().max().item()
    assert diff <= 1e-5, f"ExpertBlock vs dense reference: max abs diff {diff:.3e}"
    print(f"\nExpertBlock vs dense reference: max abs diff {diff:.3e}")


def test_prefix_kv_is_not_re_roped():
    """Shifting the prefix content must change the output; roping is applied ONLY to suffix q/k.

    If the block (wrongly) re-roped the prefix K, a prefix K built from already-roped values would
    be double-roped and the dense reference (which never ropes the prefix) would disagree — covered
    above. Here we sanity-check sensitivity: attention output actually depends on prefix K/V.
    """
    block = ExpertBlock(CFG)
    for m in block.modules():
        if isinstance(m, AdaRMSNorm):
            nn.init.normal_(m.mod.weight, std=0.2)
            nn.init.normal_(m.mod.bias, std=0.2)
    x = torch.randn(B, T, CFG.width)
    cond = torch.randn(B, CFG.width)
    pos_emb = _rope_tables(torch.arange(100, 100 + T), CFG.head_dim)
    k_pre = torch.randn(B, CFG.num_kv_heads, S_P, CFG.head_dim)
    v_pre = torch.randn(B, CFG.num_kv_heads, S_P, CFG.head_dim)
    with torch.no_grad():
        a = block(x, k_pre, v_pre, pos_emb, cond)
        b = block(x, k_pre, v_pre + 1.0, pos_emb, cond)
    assert (a - b).abs().max() > 1e-4, "expert output ignores the prefix V — wiring broken"
