"""Real early-exit through the Qwen2.5-VL text model.

This adapts ``Qwen2_5_VLTextModel.forward`` (transformers 4.57.6, modeling_qwen2_5_vl.py L791-923):
we reuse the official mask builders, rotary embedding, and per-layer call signature, but iterate only
``layers[:num_layers]`` so fewer layers actually run (genuine compute savings — NOT post-hoc indexing
of a full 36-layer pass).

Indexing convention (verified against the source loop):
- The official ``output_hidden_states`` tuple stores, at index i, the INPUT to layer i (== output of
  layer i-1), and appends the post-final-norm tensor last (length 37 for 36 layers).
- Therefore ``run_language_early_exit(..., num_layers=N, early_exit_norm="none")`` runs layers
  0..N-1 and returns the output of layer N-1, which equals ``full.hidden_states[N]`` (pre-norm).
- ``num_layers=36, early_exit_norm="final_norm"`` applies the final norm and equals
  ``full.last_hidden_state``.
"""

from __future__ import annotations

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


@torch.inference_mode()
def run_language_early_exit(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    num_layers: int,
    early_exit_norm: str = "final_norm",
) -> torch.Tensor:
    """Run the first ``num_layers`` decoder layers on ``inputs_embeds``.

    Args:
        model: Qwen2_5_VLForConditionalGeneration (we use model.model.language_model).
        inputs_embeds: [B, S, 2048] already-merged embeddings (video scattered in + state appended).
        attention_mask: [B, S] of 1s/0s (NOT a 4D/packed mask; we build the causal mask from it).
        position_ids: [3, B, S] mrope positions from get_rope_index (+ state extension).
        num_layers: number of decoder layers to run (1..36).
        early_exit_norm: "final_norm" applies lm.norm at the end; "none" returns pre-norm.
    """
    lm = model.model.language_model
    if not (1 <= num_layers <= len(lm.layers)):
        raise ValueError(f"num_layers must be in [1, {len(lm.layers)}], got {num_layers}")

    B, S, _ = inputs_embeds.shape
    cache_position = torch.arange(S, device=inputs_embeds.device)

    # Mirror the official mask construction (position_ids=None: we pass a normal 2D attention_mask,
    # so the model is NOT in packed-FA2 mode and text_position_ids is None there too).
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

    position_embeddings = lm.rotary_emb(inputs_embeds, position_ids)

    hidden_states = inputs_embeds
    for decoder_layer in lm.layers[:num_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=masks[decoder_layer.attention_type],
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]

    if early_exit_norm == "final_norm":
        hidden_states = lm.norm(hidden_states)
    elif early_exit_norm != "none":
        raise ValueError(f"early_exit_norm must be 'final_norm' or 'none', got {early_exit_norm!r}")
    return hidden_states
