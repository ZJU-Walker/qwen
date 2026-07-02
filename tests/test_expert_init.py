"""T6: expert init sanity — zero-init adaRMS gates make every block an exact identity.

At init the full-size expert must compute out == action_out_proj(rms_norm(action_in_proj(x)))
for any tau and any prefix K/V (the prefix is invisible until the gates learn to open), with
finite outputs and ~850M params.
"""

import torch

from streaming_qwen_vlm.expert import ActionExpert, ExpertConfig

B, S_P = 2, 40


def test_expert_init_identity_and_param_count():
    torch.manual_seed(0)
    cfg = ExpertConfig()
    expert = ActionExpert(cfg)

    n_params = sum(p.numel() for p in expert.parameters())
    assert 0.5e9 < n_params < 1.2e9, f"unexpected expert size: {n_params/1e6:.0f}M"
    print(f"\nActionExpert params: {n_params/1e6:.0f}M")

    x = torch.randn(B, cfg.horizon, cfg.action_dim)
    prefix_kv = [
        (torch.randn(B, cfg.num_kv_heads, S_P, cfg.head_dim),
         torch.randn(B, cfg.num_kv_heads, S_P, cfg.head_dim))
        for _ in range(cfg.depth)
    ]
    pos = torch.arange(2000, 2000 + cfg.horizon)
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
    freqs = pos.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], -1)
    pos_emb = (emb.cos()[None, None].expand(3, B, cfg.horizon, cfg.head_dim).contiguous(),
               emb.sin()[None, None].expand(3, B, cfg.horizon, cfg.head_dim).contiguous())

    for tau_val in (0.001, 0.5, 1.0):
        tau = torch.full((B,), tau_val)
        with torch.no_grad():
            out = expert(x, tau, prefix_kv, pos_emb)
            # zero gates -> identity blocks -> only in_proj + final (plain) RMS + out_proj act
            h = expert.action_in_proj(x)
            hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_eps)
            expected = expert.action_out_proj(hn)
        assert torch.isfinite(out).all(), f"non-finite expert output at tau={tau_val}"
        diff = (out - expected).abs().max().item()
        assert diff <= 1e-5, f"tau={tau_val}: expert not identity at init (max abs diff {diff:.3e})"

    # ...and the prefix really is invisible at init (gates closed).
    with torch.no_grad():
        out2 = expert(x, torch.full((B,), 0.5),
                      [(k + 3.0, v - 3.0) for k, v in prefix_kv], pos_emb)
    assert (out - out2).abs().max().item() <= 1e-6
