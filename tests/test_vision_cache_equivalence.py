"""Check #2 (GATING): per-pair encoding concatenated == full-window encoding.

The pair cache concatenates 15 independently-encoded 2-frame embeddings and treats that as if it were
the feature of one 30-frame video. That is only valid if the vision tower does NOT mix information
across temporal pairs. We verify it empirically in fp32 with a tight tolerance. On failure we raise a
detailed diagnostic and STOP — there is no fallback (a silent full-window recompute would defeat the
entire caching design).
"""

import contextlib

import torch

from streaming_qwen_vlm.preprocess import (
    frames_to_video_inputs,
    pair_to_pixel_values,
    prepare_frames,
)
from .conftest import make_frames

# Tolerances tuned on the first H200 run (recorded in benchmark/results.md):
#   observed fp32: max abs diff ~1.4e-4, mean abs diff ~8e-7.
# The MEAN being ~1e-6 is the strong proof of equivalence: genuine cross-pair attention mixing would
# raise the mean by several orders of magnitude (cf. the bf16 mean ~2.6e-2 from precision alone). The
# ~1.4e-4 MAX is localized fp32 attention roundoff (SDPA chunked-per-pair vs one full pass).
FP32_MAX_TOL = 1e-3
FP32_MEAN_TOL = 1e-5


@contextlib.contextmanager
def _visual_in_fp32(model):
    """Cast the vision tower to fp32 for a tight check. FlashAttention-2 rejects fp32, so we also
    swap every vision module's attention implementation to SDPA, restoring both afterwards."""
    visual = model.model.visual
    orig_dtype = next(visual.parameters()).dtype
    # Collect every config object under the vision tower that carries an attn-impl flag.
    configs = []
    for mod in visual.modules():
        cfg_obj = getattr(mod, "config", None)
        if cfg_obj is not None and hasattr(cfg_obj, "_attn_implementation"):
            configs.append((cfg_obj, cfg_obj._attn_implementation))
    if hasattr(visual, "config") and hasattr(visual.config, "_attn_implementation"):
        configs.append((visual.config, visual.config._attn_implementation))

    visual.to(torch.float32)
    for cfg_obj, _ in configs:
        cfg_obj._attn_implementation = "sdpa"
    try:
        yield
    finally:
        visual.to(orig_dtype)
        for cfg_obj, orig in configs:
            cfg_obj._attn_implementation = orig


def _encode_full(model, processor, prepared, cfg):
    inp = frames_to_video_inputs(processor, prepared, cfg)
    pv = inp["pixel_values_videos"].to(model.device)
    grid = inp["video_grid_thw"].to(model.device)
    return model.get_video_features(pv, grid)[0]  # [2160, 2048]


def _encode_pairs(model, processor, prepared, cfg):
    embeds = []
    for i in range(cfg.num_pairs):
        two = prepared[2 * i : 2 * i + 2]
        inp = pair_to_pixel_values(processor, two, cfg)
        pv = inp["pixel_values_videos"].to(model.device)
        grid = inp["video_grid_thw"].to(model.device)
        embeds.append(model.get_video_features(pv, grid)[0])  # [144, 2048]
    return torch.cat(embeds, dim=0)  # [2160, 2048]


def test_vision_cache_equivalence(cfg, model, processor):
    frames = make_frames(cfg.window_frames, seed=42, size=cfg.fixed_resolution)
    prepared = prepare_frames(frames, cfg)

    # bf16 reference diff (logged, not gating)
    full_bf16 = _encode_full(model, processor, prepared, cfg).float()
    pairs_bf16 = _encode_pairs(model, processor, prepared, cfg).float()
    bf16_max = (full_bf16 - pairs_bf16).abs().max().item()
    bf16_mean = (full_bf16 - pairs_bf16).abs().mean().item()

    # fp32 tight check (gating)
    with _visual_in_fp32(model):
        full = _encode_full(model, processor, prepared, cfg)
        pairs = _encode_pairs(model, processor, prepared, cfg)

    assert full.shape == pairs.shape, (full.shape, pairs.shape)
    diff = (full - pairs).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    # first offending pair (block of tokens_per_pair rows)
    offending = -1
    per_pair = diff.view(cfg.num_pairs, cfg.tokens_per_pair, -1).amax(dim=(1, 2))
    over = (per_pair > FP32_MAX_TOL).nonzero(as_tuple=True)[0]
    if len(over):
        offending = int(over[0])

    print(
        f"\n[vision_cache_equivalence] fp32 max={max_abs:.3e} mean={mean_abs:.3e} "
        f"| bf16 max={bf16_max:.3e} mean={bf16_mean:.3e} | dtype=fp32 "
        f"shapes full={tuple(full.shape)} pairs={tuple(pairs.shape)}"
    )

    if max_abs > FP32_MAX_TOL or mean_abs > FP32_MEAN_TOL:
        raise AssertionError(
            "PAIR-CACHE EQUIVALENCE FAILED — the vision tower mixes information across temporal "
            "pairs, so the two-frame cache is NOT valid. Do NOT fall back to full-window recompute "
            "silently; redesign required.\n"
            f"  fp32 max abs diff : {max_abs:.6e} (tol {FP32_MAX_TOL:.1e})\n"
            f"  fp32 mean abs diff: {mean_abs:.6e} (tol {FP32_MEAN_TOL:.1e})\n"
            f"  bf16 max/mean     : {bf16_max:.6e} / {bf16_mean:.6e}\n"
            f"  shapes            : full={tuple(full.shape)} pairs={tuple(pairs.shape)}\n"
            f"  first offending pair index: {offending}"
        )
