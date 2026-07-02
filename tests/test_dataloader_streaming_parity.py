"""T2 (GATING): the training dataloader and the streaming path see the same inputs.

For a real episode and several control steps t (including front-padded early ticks):
- pair frame indices + processed pixel tensors are BIT-EXACT between TrossenActDataset and a
  simulated streaming feed (both run the same prepare_frames + pair_to_pixel_values);
- window features agree between the dataset's batched ViT call (all 15 pair grid rows at once)
  and the streaming PairVisionCache sequential encode + front-padded concat, to bf16 kernel
  tolerance (batched vs sequential GEMMs; the blocked vision attention makes them structurally
  equivalent — Stage-1 test #2);
- state bins and the hold-pose-padded action chunk match a hand computation.

Needs the real model (H200) + the Trossen dataset. The FAST tokenizer is stubbed out — this test
is about vision/state/action parity, not FAST (see test_fast_roundtrip.py).
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from streaming_qwen_vlm.normalize import compute_norm_stats, discretize, load_episode_arrays
from streaming_qwen_vlm.preprocess import pair_to_pixel_values, prepare_frames
from streaming_qwen_vlm.state_text import bins_to_ids, build_state_lut
from streaming_qwen_vlm.training.dataset import (
    FRAME_STRIDE,
    TICK_STRIDE,
    TrossenActDataset,
    pair_frame_indices,
)
from streaming_qwen_vlm.vision_cache import PairVisionCache

ROOT = "/iris/projects/humanoid/trossen_data/0528_merge_block_mem"
EPISODE = 0
T_VALUES = [10, 25, 50, 131]  # includes heavily front-padded early ticks
FEATURE_TOL = 2e-2            # bf16 batched-vs-sequential kernel noise
MAX_FAST = 16

pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason=f"dataset not found at {ROOT}")


class _FakeFast:
    """Stub: T2 must not require the FAST download."""

    def encode_padded(self, chunk, max_tokens):
        return np.zeros(max_tokens, dtype=np.int64), 1


@pytest.fixture(scope="module")
def dataset(cfg, processor):
    stats = compute_norm_stats(ROOT, [EPISODE])
    ds = TrossenActDataset(
        root=ROOT, episodes=[EPISODE], cfg=cfg, processor=processor,
        state_lut=build_state_lut(processor.tokenizer),
        action_stats=stats["action"], state_stats=stats["state"],
        max_fast_tokens=MAX_FAST,
    )
    ds._fast = _FakeFast()
    return ds


def _load_frame(ep: int, idx: int) -> np.ndarray:
    path = os.path.join(ROOT, "frames", "cam_high", f"episode_{ep:06d}", f"frame_{idx:06d}.jpg")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


@pytest.mark.parametrize("t", T_VALUES)
def test_pixel_and_lowdim_parity(dataset, cfg, processor, t):
    item = dataset[dataset.samples.index((EPISODE, t))]

    # --- pixels: bit-exact vs an independent rebuild over the streaming pair grid ---
    ks = pair_frame_indices(t, cfg.num_pairs)
    pvs = {}
    for k in dict.fromkeys(ks):
        prepared = prepare_frames(
            [_load_frame(EPISODE, TICK_STRIDE * k), _load_frame(EPISODE, TICK_STRIDE * k + FRAME_STRIDE)],
            cfg,
        )
        pvs[k] = pair_to_pixel_values(processor, prepared, cfg)["pixel_values_videos"]
    expected = torch.cat([pvs[k] for k in ks], dim=0)
    assert torch.equal(item["pixel_values"], expected), f"t={t}: pixel tensors differ"

    # --- state bins ---
    arrs = load_episode_arrays(ROOT, EPISODE)
    bins = discretize(arrs["state"][t], dataset.state_stats, bins=cfg.state_bins)
    assert torch.equal(item["state_ids"], bins_to_ids(bins, dataset.state_lut)), f"t={t}: state ids"

    # --- action chunk: hold-pose padding ---
    chunk = arrs["action"][t : t + dataset.horizon]
    if len(chunk) < dataset.horizon:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], dataset.horizon - len(chunk), 0)])
    from streaming_qwen_vlm.normalize import normalize

    np.testing.assert_allclose(item["actions_norm"].numpy(),
                               normalize(chunk, dataset.action_stats), atol=1e-6)


@pytest.mark.parametrize("t", [10, 131])
def test_feature_parity_batched_vs_streaming(dataset, cfg, model, t):
    item = dataset[dataset.samples.index((EPISODE, t))]
    device = model.device

    # Streaming: sequential per-pair encode + front-padded concat (the deployment path).
    cache = PairVisionCache(cfg.num_pairs, cfg.tokens_per_pair)
    k_max = (t - 10) // TICK_STRIDE
    per_pair = item["pixel_values"].view(cfg.num_pairs, -1, item["pixel_values"].shape[-1])
    grid1 = torch.tensor([list(cfg.pair_grid_thw)], dtype=torch.long)
    seen = set()
    for k, pv in zip(pair_frame_indices(t, cfg.num_pairs), per_pair):
        if k in seen:
            continue
        seen.add(k)
        cache.push(cache.encode_pair(model, pv, grid1).to(model.dtype))
    assert len(cache) == min(k_max + 1, cfg.num_pairs)
    streamed = cache.concat(pad_to_full=True)  # [2160, 2048]

    # Dataset/training: ONE batched ViT call over all 15 grid rows.
    with torch.inference_mode():
        grid = torch.tensor([list(cfg.pair_grid_thw)] * cfg.num_pairs, dtype=torch.long)
        feats = model.get_video_features(item["pixel_values"].to(device), grid.to(device))
        batched = torch.cat(list(feats), dim=0)

    diff = (streamed.float() - batched.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        streamed.float().flatten(), batched.float().flatten(), dim=0
    ).item()
    print(f"\nt={t}: batched vs streaming features: max abs diff {diff:.3e}, cosine {cos:.6f}")
    assert diff <= FEATURE_TOL and cos >= 0.999, f"t={t}: diff={diff:.3e} cos={cos:.6f}"
