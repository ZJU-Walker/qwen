"""Check #6: VLMOutput shapes and token-type counts across early-exit depths."""

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

    out = None
    for i in range(cfg.num_pairs):
        out = svlm.step([frames[2 * i], frames[2 * i + 1]], state)
        if i < cfg.num_pairs - 1:
            assert out is None  # warm-up until cache full

    assert out is not None
    instr_len = svlm.template.instr_len
    S = cfg.total_video_tokens + instr_len + cfg.num_state_tokens

    assert tuple(out.context_tokens.shape) == (1, S, 2048)
    assert tuple(out.context_mask.shape) == (1, S)
    assert tuple(out.token_types.shape) == (1, S)
    assert out.layer_index == num_layers

    tt = out.token_types[0]
    assert int((tt == TOKEN_TYPE_VIDEO).sum()) == cfg.total_video_tokens
    assert int((tt == TOKEN_TYPE_STATE).sum()) == cfg.num_state_tokens
    assert int((tt == TOKEN_TYPE_TEXT).sum()) == instr_len
    assert int(out.context_mask.sum()) == S
