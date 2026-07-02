"""Check #6: VLMOutput shapes and token-type counts across early-exit depths.

Stage-2 semantics: step() emits from the very first tick (front-padded window, D6) — there is no
warm-up None phase anymore. Default cfg has num_state_tokens == 0 (StateProjector retired).
"""

import numpy as np
import pytest

from streaming_qwen_vlm import StreamingQwenVLM, VLMConfig
from streaming_qwen_vlm.outputs import TOKEN_TYPE_STATE, TOKEN_TYPE_TEXT, TOKEN_TYPE_VIDEO
from .conftest import make_frames


@pytest.mark.parametrize("num_layers", [8, 16, 36])
def test_output_shape(model, processor, num_layers):
    cfg = VLMConfig(num_llm_layers_to_run=num_layers)
    svlm = StreamingQwenVLM(cfg, model=model, processor=processor)

    frames = make_frames(cfg.window_frames, seed=5, size=cfg.fixed_resolution)
    state = np.zeros(cfg.state_dim, dtype=np.float32)

    instr_len = svlm.template.instr_len
    S = cfg.total_video_tokens + instr_len + cfg.num_state_tokens

    for i in range(cfg.num_pairs):
        out = svlm.step([frames[2 * i], frames[2 * i + 1]], state)
        # Front-padding: a full-shape output at EVERY tick, including the very first.
        assert out is not None, f"step {i}: expected output from tick 0 (front-padding)"
        assert tuple(out.context_tokens.shape) == (1, S, 2048)

    assert tuple(out.context_mask.shape) == (1, S)
    assert tuple(out.token_types.shape) == (1, S)
    assert out.layer_index == num_layers

    tt = out.token_types[0]
    assert int((tt == TOKEN_TYPE_VIDEO).sum()) == cfg.total_video_tokens
    assert int((tt == TOKEN_TYPE_STATE).sum()) == cfg.num_state_tokens
    assert int((tt == TOKEN_TYPE_TEXT).sum()) == instr_len
    assert int(out.context_mask.sum()) == S
