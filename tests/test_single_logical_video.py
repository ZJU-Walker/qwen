"""Check #1: the rolling window is ONE logical video, not 15 separate videos."""

from streaming_qwen_vlm.config import VIDEO_TOKEN_ID
from streaming_qwen_vlm.prompt_builder import build_prompt


def test_single_logical_video(cfg, processor):
    tmpl = build_prompt(processor, cfg)

    assert list(tmpl.video_grid_thw.shape) == [1, 3], tmpl.video_grid_thw.shape
    assert int(tmpl.video_grid_thw[0, 0]) == cfg.num_pairs  # temporal == 15 (single row)
    assert int(tmpl.video_grid_thw[0, 1]) == cfg.grid_h
    assert int(tmpl.video_grid_thw[0, 2]) == cfg.grid_w

    n_video = int((tmpl.input_ids == VIDEO_TOKEN_ID).sum())
    assert n_video == cfg.total_video_tokens == 2160, n_video
