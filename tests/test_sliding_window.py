"""Check #5: the pair cache shifts correctly — after overflow it holds the latest 15 pairs in order.

Frames are sentinel-tagged (frame i is a constant plane of value i) so each pair embedding is
identifiable by re-encoding the expected pair and matching it to the cached slot. We assert the cached
embeddings occupy the correct current slots; we do NOT assert position_ids change (the template and
positions are fixed by design).
"""

from streaming_qwen_vlm import StreamingQwenVLM, VLMConfig
from streaming_qwen_vlm.preprocess import pair_to_pixel_values, prepare_frames
from .conftest import tagged_frames


def test_sliding_window(model, processor):
    cfg = VLMConfig()
    svlm = StreamingQwenVLM(cfg, model=model, processor=processor)

    n_pairs_fed = cfg.num_pairs + 1  # one overflow -> oldest pair evicted
    n_frames = cfg.frames_per_pair * n_pairs_fed
    frames = tagged_frames(n_frames, size=cfg.fixed_resolution)

    for i in range(n_pairs_fed):
        svlm._encode_and_push_pair([frames[2 * i], frames[2 * i + 1]])

    assert svlm.cache.is_full()
    assert len(svlm.cache) == cfg.num_pairs

    # After one overflow the cache holds pairs starting at frame index 2 (i.e. (f3,f4)...(f31,f32)).
    for slot in range(cfg.num_pairs):
        fa = cfg.frames_per_pair * (slot + 1)
        prepared = prepare_frames([frames[fa], frames[fa + 1]], cfg)
        pinp = pair_to_pixel_values(processor, prepared, cfg)
        fresh = svlm.cache.encode_pair(
            model, pinp["pixel_values_videos"], pinp["video_grid_thw"]
        ).to(svlm.dtype)
        cached = svlm.cache._buf[slot]
        max_abs = (fresh.float() - cached.float()).abs().max().item()
        assert max_abs <= 1e-3, f"slot {slot}: cached pair mismatch, max abs diff {max_abs:.3e}"
