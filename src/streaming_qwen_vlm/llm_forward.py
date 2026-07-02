"""Train-capable manual pass through the Qwen2.5-VL decoder with per-layer prefix K/V export.

This is the Stage-2 seam the action expert hangs off (claude_plan D1/D2). For every decoder layer
we recompute the layer's post-rope K/V over the PREFIX region from the same hidden states the layer
receives, under torch.no_grad() — that no_grad IS the knowledge-insulation stop-gradient boundary:
the backbone trains only through the autoregressive path, never through the expert's flow loss.

Why recompute instead of DynamicCache / hooks:
- gradient checkpointing force-disables use_cache inside checkpointed layers;
- K/V are locals inside the attention forward, invisible to module hooks;
- recompute costs 2 small GEMMs/layer over the prefix and is equivalence-testable: in the stock
  attention (modeling_qwen2_5_vl.py, v4.57.6) K is roped BEFORE past_key_values.update, so a
  recomputed post-rope K equals stock cache contents exactly (tests/test_kv_export_equivalence.py).

``early_exit.py`` (Stage 1) is left untouched and remains the ablation path.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

# 36 x (K, V), each [B, num_kv_heads=2, S_prefix, head_dim=128], post-rope, detached.
PrefixKV = List[Tuple[torch.Tensor, torch.Tensor]]


def _build_masks(lm, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                 cache_position: torch.Tensor) -> dict:
    """Same construction as early_exit.py / the official Qwen2_5_VLTextModel.forward."""
    mask_kwargs = dict(
        config=lm.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=None,
    )
    masks = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(lm, "has_sliding_layers", False):
        masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    return masks


def _export_layer_kv(layer, h_prefix: torch.Tensor, cos_p: torch.Tensor, sin_p: torch.Tensor,
                     mrope_section, num_kv_heads: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute this layer's post-rope prefix (K, V) from its input hidden states.

    Replicates the stock attention path exactly: input_layernorm -> k_proj/v_proj (bias=True picked
    up from the modules) -> [B, kv_heads, S_p, head_dim] -> multimodal rope on K. Call under
    torch.no_grad(); prefix K is NEVER re-roped downstream.
    """
    B, S_p, _ = h_prefix.shape
    normed = layer.input_layernorm(h_prefix)
    k = layer.self_attn.k_proj(normed).view(B, S_p, num_kv_heads, head_dim).transpose(1, 2)
    v = layer.self_attn.v_proj(normed).view(B, S_p, num_kv_heads, head_dim).transpose(1, 2)
    k, _ = apply_multimodal_rotary_pos_emb(k, k, cos_p, sin_p, mrope_section)
    return k, v


def forward_train(
    model,
    inputs_embeds: torch.Tensor,      # [B, S_total, 2048] video-scattered prefix + FAST tail
    position_ids: torch.Tensor,       # [3, B, S_total] mrope positions
    S_prefix: int,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PrefixKV, torch.Tensor]:
    """Run all 36 layers with grads (call under torch.autocast); export detached prefix K/V.

    Returns (prefix_kv, hidden_states after the final norm — feed to lm_head for the AR loss).
    Per-layer gradient checkpointing works untouched: layers are GradientCheckpointingLayer and
    self-checkpoint when model.gradient_checkpointing_enable() was called and the model is training.

    MUST run under ``torch.autocast(..., cache_enabled=False)`` when training: the no_grad KV
    export below calls k_proj/v_proj/input_layernorm before each layer's forward, and with the
    autocast weight cache enabled those casts (created under no_grad, hence grad-disconnected)
    would be reused by the layer itself — CheckpointError on backward and silently missing
    k/v/norm gradients. Verified by minimal repro; the guard below makes the failure actionable.
    """
    if (
        torch.is_grad_enabled()
        and torch.is_autocast_enabled("cuda")
        and torch.is_autocast_cache_enabled()
    ):
        raise RuntimeError(
            "forward_train requires torch.autocast(..., cache_enabled=False): the no_grad KV "
            "export would poison the autocast weight cache with grad-disconnected casts "
            "(CheckpointError in backward; k/v/norm weights get no gradient)."
        )
    lm = model.model.language_model
    cfg = lm.config
    mrope_section = cfg.rope_scaling["mrope_section"]
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    B, S, _ = inputs_embeds.shape
    if not (0 < S_prefix <= S):
        raise ValueError(f"S_prefix={S_prefix} out of range for S={S}")
    device = inputs_embeds.device
    cache_position = torch.arange(S, device=device)
    attention_mask = torch.ones(B, S, dtype=torch.long, device=device)
    masks = _build_masks(lm, inputs_embeds, attention_mask, cache_position)

    cos, sin = lm.rotary_emb(inputs_embeds, position_ids)
    # Pin the rope tables (and therefore the exported K) to the compute dtype: with fp32 master
    # params + bf16 autocast, rotary_emb returns fp32 and would silently promote K/attention inputs.
    cos, sin = cos.to(compute_dtype), sin.to(compute_dtype)
    cos_p, sin_p = cos[:, :, :S_prefix], sin[:, :, :S_prefix]

    prefix_kv: PrefixKV = []
    h = inputs_embeds
    for layer in lm.layers:
        with torch.no_grad():  # knowledge insulation: the expert never backprops into the backbone
            prefix_kv.append(
                _export_layer_kv(layer, h[:, :S_prefix], cos_p, sin_p, mrope_section,
                                 num_kv_heads, head_dim)
            )
        h = layer(
            h,
            attention_mask=masks[layer.attention_type],
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=(cos, sin),
        )[0]

    return prefix_kv, lm.norm(h)


@torch.no_grad()
def forward_prefill(
    model,
    inputs_embeds: torch.Tensor,      # [B, S_prefix, 2048] prefix only (no FAST tail at inference)
    position_ids: torch.Tensor,       # [3, B, S_prefix]
) -> PrefixKV:
    """Inference twin of forward_train: export per-layer prefix K/V, no AR head.

    The LAST layer's output is never consumed (K/V come from layer INPUTS), so its forward is
    skipped — the expert's denoise loop then reuses this PrefixKV for all Euler steps.
    """
    lm = model.model.language_model
    cfg = lm.config
    mrope_section = cfg.rope_scaling["mrope_section"]
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    B, S, _ = inputs_embeds.shape
    device = inputs_embeds.device
    cache_position = torch.arange(S, device=device)
    attention_mask = torch.ones(B, S, dtype=torch.long, device=device)
    masks = _build_masks(lm, inputs_embeds, attention_mask, cache_position)

    cos, sin = lm.rotary_emb(inputs_embeds, position_ids)
    cos, sin = cos.to(inputs_embeds.dtype), sin.to(inputs_embeds.dtype)

    prefix_kv: PrefixKV = []
    h = inputs_embeds
    last = len(lm.layers) - 1
    for i, layer in enumerate(lm.layers):
        prefix_kv.append(
            _export_layer_kv(layer, h, cos, sin, mrope_section, num_kv_heads, head_dim)
        )
        if i < last:
            h = layer(
                h,
                attention_mask=masks[layer.attention_type],
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
            )[0]

    return prefix_kv
