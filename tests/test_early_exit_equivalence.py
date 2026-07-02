"""Check #4: real early exit matches the official text-model forward.

Indexing convention (see early_exit.py): the official output_hidden_states tuple stores the INPUT to
layer i at index i and appends the post-final-norm tensor last. So:
  - run_language_early_exit(..., N, "none")  == full.hidden_states[N]   (pre-norm, output of layer N-1)
  - run_language_early_exit(..., 36, "final_norm") == full.last_hidden_state
"""

import torch

from streaming_qwen_vlm.config import NUM_LLM_LAYERS
from streaming_qwen_vlm.early_exit import run_language_early_exit

PRENORM_TOL = 5e-3


def _make_inputs(model, cfg, S=48):
    lm = model.model.language_model
    g = torch.Generator(device="cpu").manual_seed(7)
    embeds = torch.randn(1, S, 2048, generator=g).to(model.device, dtype=next(lm.parameters()).dtype)
    attn = torch.ones(1, S, device=model.device, dtype=torch.long)
    pos = torch.arange(S, device=model.device).view(1, 1, S).expand(3, 1, S).contiguous()
    return embeds, attn, pos


def test_early_exit_prenorm(cfg, model):
    embeds, attn, pos = _make_inputs(model, cfg)
    lm = model.model.language_model
    full = lm(
        inputs_embeds=embeds,
        attention_mask=attn,
        position_ids=pos,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    # hidden_states has length 37 (36 layer-inputs + final post-norm).
    assert len(full.hidden_states) == NUM_LLM_LAYERS + 1, len(full.hidden_states)

    for n in (8, 16):
        he = run_language_early_exit(model, embeds, attn, pos, num_layers=n, early_exit_norm="none")
        ref = full.hidden_states[n]
        max_abs = (he.float() - ref.float()).abs().max().item()
        print(f"\n[early_exit N={n} none] max abs diff = {max_abs:.3e}")
        assert max_abs <= PRENORM_TOL, f"N={n}: max abs diff {max_abs:.3e} > {PRENORM_TOL:.1e}"


def test_early_exit_full_with_final_norm(cfg, model):
    embeds, attn, pos = _make_inputs(model, cfg)
    lm = model.model.language_model
    full = lm(
        inputs_embeds=embeds,
        attention_mask=attn,
        position_ids=pos,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    he = run_language_early_exit(
        model, embeds, attn, pos, num_layers=NUM_LLM_LAYERS, early_exit_norm="final_norm"
    )
    max_abs = (he.float() - full.last_hidden_state.float()).abs().max().item()
    print(f"\n[early_exit N=36 final_norm vs last_hidden_state] max abs diff = {max_abs:.3e}")
    assert max_abs <= PRENORM_TOL, f"max abs diff {max_abs:.3e} > {PRENORM_TOL:.1e}"
