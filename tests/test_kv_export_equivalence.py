"""T1 (GATING): exported per-layer prefix K/V == the stock use_cache forward's cache contents.

The stock Qwen2.5-VL attention ropes K BEFORE past_key_values.update (modeling_qwen2_5_vl.py,
v4.57.6), so llm_forward's recomputed post-rope K/V must match a DynamicCache bit-for-bit up to
kernel batching noise. Also checks forward_train's final hidden states against the official
forward, and that forward_train == forward_prefill on the K/V they both export.
"""

import pytest
import torch

from streaming_qwen_vlm.config import NUM_LLM_LAYERS
from streaming_qwen_vlm.llm_forward import forward_prefill, forward_train

TOL = 2e-3  # bf16
S = 48


def _cache_layer_kv(cache, i):
    """DynamicCache access across transformers versions."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache[i], cache.value_cache[i]
    layer = cache.layers[i]
    return layer.keys, layer.values


def _make_inputs(model):
    torch.manual_seed(7)
    embeds = torch.randn(1, S, 2048).to(model.device, model.dtype)
    pos = torch.arange(S).view(1, 1, S).expand(3, 1, S).to(model.device)
    return embeds, pos


def test_prefill_kv_matches_stock_cache(model):
    lm = model.model.language_model
    embeds, pos = _make_inputs(model)

    kv = forward_prefill(model, embeds, pos)
    assert len(kv) == NUM_LLM_LAYERS

    full = lm(
        inputs_embeds=embeds,
        attention_mask=torch.ones(1, S, dtype=torch.long, device=model.device),
        position_ids=pos,
        use_cache=True,
        return_dict=True,
    )
    cache = full.past_key_values

    worst_k = worst_v = 0.0
    for i, (k, v) in enumerate(kv):
        k_ref, v_ref = _cache_layer_kv(cache, i)
        assert k.shape == k_ref.shape == (1, 2, S, 128), f"layer {i}: {k.shape} vs {k_ref.shape}"
        dk = (k.float() - k_ref.float()).abs().max().item()
        dv = (v.float() - v_ref.float()).abs().max().item()
        worst_k, worst_v = max(worst_k, dk), max(worst_v, dv)
        assert dk <= TOL, f"layer {i}: K max abs diff {dk:.3e} > {TOL}"
        assert dv <= TOL, f"layer {i}: V max abs diff {dv:.3e} > {TOL}"
    print(f"\nKV export vs stock cache over {NUM_LLM_LAYERS} layers: "
          f"max |dK|={worst_k:.3e}, max |dV|={worst_v:.3e}")


def test_forward_train_matches_prefill_and_official(model):
    lm = model.model.language_model
    embeds, pos = _make_inputs(model)

    with torch.no_grad():
        kv_train, hidden = forward_train(model, embeds, pos, S_prefix=S,
                                         compute_dtype=model.dtype)
    kv_pre = forward_prefill(model, embeds, pos)
    for i, ((kt, vt), (kp, vp)) in enumerate(zip(kv_train, kv_pre)):
        assert torch.equal(kt, kp) and torch.equal(vt, vp), f"layer {i}: train/prefill KV differ"

    full = lm(
        inputs_embeds=embeds,
        attention_mask=torch.ones(1, S, dtype=torch.long, device=model.device),
        position_ids=pos,
        use_cache=False,
        return_dict=True,
    )
    dh = (hidden.float() - full.last_hidden_state.float()).abs().max().item()
    assert dh <= 5e-3, f"forward_train hidden vs official forward: max abs diff {dh:.3e}"
    print(f"\nforward_train final hidden vs official: max abs diff {dh:.3e}")
