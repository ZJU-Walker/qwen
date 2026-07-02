"""Check #3: cached-injection path == full forward with pixel_values_videos.

Both paths use the SAME full-window video features (so this isolates the injection logic: placeholder
count, embed order, masked_scatter, grid_thw, second_per_grid_ts, attention mask, get_rope_index) from
the pair-vs-full vision diff (covered by check #2).
"""

import torch

from streaming_qwen_vlm.config import VIDEO_TOKEN_ID
from streaming_qwen_vlm.preprocess import frames_to_video_inputs, prepare_frames
from streaming_qwen_vlm.prompt_builder import build_prompt
from .conftest import make_frames

INJECT_TOL = 2e-2


def test_cached_injection_equivalence(cfg, model, processor):
    tmpl = build_prompt(processor, cfg)
    input_ids = tmpl.input_ids.to(model.device)
    attn = tmpl.attention_mask.to(model.device)
    grid = tmpl.video_grid_thw.to(model.device)
    second = tmpl.second_per_grid_ts.to(model.device)

    frames = make_frames(cfg.window_frames, seed=99, size=cfg.fixed_resolution)
    prepared = prepare_frames(frames, cfg)
    vinp = frames_to_video_inputs(processor, prepared, cfg)
    pv = vinp["pixel_values_videos"].to(model.device)

    # (a) reference: full inner-model forward with pixels (vision encode + internal merge).
    ref = model.model(
        input_ids=input_ids,
        attention_mask=attn,
        pixel_values_videos=pv,
        video_grid_thw=grid,
        second_per_grid_ts=second,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    ref_last = ref.last_hidden_state

    # (b) cached-injection path: same video features, scattered into inputs_embeds, no pixels.
    video_embeds = model.get_video_features(pv, grid)[0].to(model.device)  # [2160, 2048]
    inputs_embeds = model.get_input_embeddings()(input_ids)
    vmask = (input_ids == VIDEO_TOKEN_ID).unsqueeze(-1)
    inputs_embeds = inputs_embeds.masked_scatter(
        vmask, video_embeds.to(inputs_embeds.dtype).reshape(-1)
    )
    pos, _ = model.model.get_rope_index(input_ids, None, grid, second, attn)
    out = model.model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        position_ids=pos,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    inj_last = out.last_hidden_state

    max_abs = (ref_last.float() - inj_last.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        ref_last.float().flatten(), inj_last.float().flatten(), dim=0
    ).item()
    print(f"\n[cached_injection] max abs diff = {max_abs:.3e} cosine = {cos:.6f}")
    assert max_abs <= INJECT_TOL, f"max abs diff {max_abs:.3e} > {INJECT_TOL:.1e}"
    assert cos >= 0.999, f"cosine {cos:.6f} < 0.999"
