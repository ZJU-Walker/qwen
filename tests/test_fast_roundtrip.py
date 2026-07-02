"""T4: FAST tokenize -> detokenize round-trips real action chunks within quantization error,
plus pure-CPU checks of the union-vocab AR row construction.

The round-trip tests need scipy + the physical-intelligence/fast download (M0 env step); they
skip cleanly when either is missing. Prints the tokens/chunk stats used to freeze max_fast_tokens.
"""

import os

import numpy as np
import pytest
import torch

from streaming_qwen_vlm.config import VLMConfig
from streaming_qwen_vlm.fast_tokens import (
    IGNORE_INDEX,
    IM_END_ID,
    PAD_INPUT_ID,
    V_BASE,
    FastActionTokenizer,
    build_ar_row,
)
from streaming_qwen_vlm.normalize import (
    compute_norm_stats,
    load_episode_arrays,
    load_stats,
    normalize,
)

ROOT = "/iris/projects/humanoid/trossen_data/0528_merge_block_mem"
STATS = "checkpoints/norm_stats.json"  # written by `python -m streaming_qwen_vlm.normalize`
HORIZON, DIM = 30, 14
MAX_FAST = VLMConfig().max_fast_tokens


# ------------------------------------------------------------------ pure CPU, no downloads
def test_build_ar_row_alignment():
    fast_ids = np.array([5, 9, 700], dtype=np.int64)
    inputs, targets = build_ar_row(fast_ids, n=3, max_tokens=6)

    assert inputs.tolist() == [V_BASE + 5, V_BASE + 9, V_BASE + 700] + [PAD_INPUT_ID] * 3
    # hidden[S_prefix-1] predicts fast_0 ... hidden at fast_2 predicts <|im_end|>:
    assert targets.tolist() == [V_BASE + 5, V_BASE + 9, V_BASE + 700, IM_END_ID] + [IGNORE_INDEX] * 3
    assert targets.shape == (7,)

    with pytest.raises(ValueError):
        build_ar_row(np.arange(9), n=9, max_tokens=6)


# ------------------------------------------------------------------ needs scipy + FAST download
@pytest.fixture(scope="module")
def fast():
    pytest.importorskip("scipy", reason="FAST needs scipy (pip install scipy)")
    try:
        return FastActionTokenizer(horizon=HORIZON, action_dim=DIM)
    except Exception as exc:  # not downloaded / no network
        pytest.skip(f"FAST tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def chunks():
    if not os.path.isdir(ROOT):
        pytest.skip(f"dataset not found at {ROOT}")
    # Prefer the real train-split stats (what training uses); 3-episode stats otherwise.
    if os.path.exists(STATS):
        stats = load_stats(STATS)["action"]
    else:
        stats = compute_norm_stats(ROOT, [0, 1, 2])["action"]
    out = []
    for ep in (0, 1, 2):
        actions = load_episode_arrays(ROOT, ep)["action"]
        for t in range(10, len(actions), 25):
            c = actions[t : t + HORIZON]
            if len(c) < HORIZON:
                c = np.concatenate([c, np.repeat(c[-1:], HORIZON - len(c), 0)])
            out.append(normalize(c, stats))
    return out


def test_fast_roundtrip_and_budget(fast, chunks):
    lengths, mses, max_errs = [], [], []
    for chunk in chunks:
        ids = fast.encode(chunk)
        lengths.append(len(ids))
        back = fast.decode(ids)
        assert back.shape == (HORIZON, DIM)
        mses.append(float(((back - chunk) ** 2).mean()))
        max_errs.append(float(np.abs(back - chunk).max()))

    arr = np.asarray(lengths)
    print(f"\nFAST vocab={fast.vocab_size}  chunks={len(chunks)}  tokens/chunk: "
          f"min={arr.min()} p50={int(np.median(arr))} p99={int(np.percentile(arr, 99))} max={arr.max()}")
    print(f"round-trip: mean MSE={np.mean(mses):.5f}  worst max-err={np.max(max_errs):.4f}")

    assert arr.max() <= MAX_FAST, (
        f"chunk needs {arr.max()} tokens > max_fast_tokens={MAX_FAST}; raise the config budget")
    # DCT+BPE quantization on [-1,1]-normalized chunks should be small on this low-motion data.
    assert np.mean(mses) <= 1e-2, f"round-trip MSE too high: {np.mean(mses):.5f}"
    assert np.max(max_errs) <= 0.5, f"round-trip max err too high: {np.max(max_errs):.4f}"


def test_encode_padded(fast, chunks):
    ids, n = fast.encode_padded(chunks[0], MAX_FAST)
    assert ids.shape == (MAX_FAST,) and 0 < n <= MAX_FAST
    assert (ids[n:] == 0).all()
    back = fast.decode(ids[:n].tolist())
    assert back.shape == (HORIZON, DIM)
    inputs, targets = build_ar_row(ids, n, MAX_FAST)
    assert int(targets[n]) == IM_END_ID
    assert (torch.as_tensor(inputs[:n]) >= V_BASE).all()
