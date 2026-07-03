"""T2 (GATING): the training dataloader and the streaming path see the same inputs.

For a real episode and several control steps t (including front-padded early ticks):
- pair frame indices + processed pixel tensors are BIT-EXACT between TrossenActDataset and a
  simulated streaming feed (both run the same prepare_frames + pair_to_pixel_values);
- window features agree between the dataset's batched ViT call (all 15 pair grid rows at once)
  and the streaming PairVisionCache sequential encode + front-padded concat. GATED IN FP32
  (mirrors test_vision_cache_equivalence): bf16 kernel accumulation legitimately produces
  max-abs diffs of O(1) on these large-magnitude features at cosine ~0.999, so bf16 numbers are
  logged, not asserted;
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
    MIN_T,
    TICK_STRIDE,
    TrossenActDataset,
    pair_frame_indices,
)
from streaming_qwen_vlm.vision_cache import PairVisionCache
from .test_vision_cache_equivalence import _visual_in_fp32

ROOT = "/iris/projects/humanoid/trossen_data/0528_merge_block_mem"
EPISODE = 0
T_VALUES = [10, 25, 50, 131]  # includes heavily front-padded early ticks
# fp32 gating tolerances, same as test_vision_cache_equivalence (observed fp32 max ~1.4e-4)
FP32_MAX_TOL = 1e-3
FP32_MEAN_TOL = 1e-5
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

    # --- pixels: dataset stores UNIQUE pairs + slot_map; the slot expansion must be bit-exact
    # vs an independent rebuild over the streaming pair grid (duplicates included) ---
    ks = pair_frame_indices(t, cfg.num_pairs)
    unique_ks = list(dict.fromkeys(ks))
    pvs = {}
    for k in unique_ks:
        prepared = prepare_frames(
            [_load_frame(EPISODE, TICK_STRIDE * k), _load_frame(EPISODE, TICK_STRIDE * k + FRAME_STRIDE)],
            cfg,
        )
        pvs[k] = pair_to_pixel_values(processor, prepared, cfg)["pixel_values_videos"]
    rows = cfg.grid_h * cfg.grid_w
    assert item["pixel_values"].shape[0] == len(unique_ks) * rows, (
        f"t={t}: a duplicate pair got re-encoded ({item['pixel_values'].shape[0]} rows, "
        f"{len(unique_ks)} unique pairs)")
    assert item["slot_map"].tolist() == [unique_ks.index(k) for k in ks], f"t={t}: slot_map wrong"
    assert torch.equal(item["pixel_values"], torch.cat([pvs[k] for k in unique_ks], dim=0)), (
        f"t={t}: unique pixel tensors differ")
    expanded = item["pixel_values"].view(len(unique_ks), rows, -1)[item["slot_map"]]
    expected = torch.cat([pvs[k] for k in ks], dim=0)
    assert torch.equal(expanded.reshape(expected.shape), expected), f"t={t}: slot expansion differs"

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
    """GATING (fp32): dataset's one batched ViT call == streaming sequential encode + front-pad."""
    item = dataset[dataset.samples.index((EPISODE, t))]
    device = model.device
    k_max = (t - MIN_T) // TICK_STRIDE
    rows = cfg.grid_h * cfg.grid_w
    n_unique = item["pixel_values"].shape[0] // rows
    per_pair = item["pixel_values"].view(n_unique, rows, -1)
    grid1 = torch.tensor([list(cfg.pair_grid_thw)], dtype=torch.long)
    grid_all = torch.tensor([list(cfg.pair_grid_thw)] * n_unique, dtype=torch.long)

    @torch.inference_mode()
    def encode_both():
        # Streaming: sequential per-pair encode + front-padded concat (the deployment path).
        cache = PairVisionCache(cfg.num_pairs, cfg.tokens_per_pair)
        for pv in per_pair:  # dataset uniques are chronological == streaming push order
            cache.push(cache.encode_pair(model, pv, grid1))  # keep the tower's compute dtype
        assert len(cache) == min(k_max + 1, cfg.num_pairs) == n_unique
        streamed = cache.concat(pad_to_full=True)  # [2160, 2048]
        # Dataset/training: ONE batched ViT call over the unique pairs, slot-gathered.
        feats = model.get_video_features(item["pixel_values"].to(device), grid_all.to(device))
        feats = torch.stack(list(feats))  # [n_unique, 144, 2048]
        batched = feats[item["slot_map"].to(device)].reshape(cfg.total_video_tokens, -1)
        return streamed, batched

    # bf16 diff is logged only: kernel accumulation gives max-abs O(1) at cosine ~0.999 on these
    # large-magnitude features (cf. Stage-1 test #2: bf16 max ~7.8 vs fp32 max ~1.4e-4).
    s16, b16 = encode_both()
    bf16_max = (s16.float() - b16.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        s16.float().flatten(), b16.float().flatten(), dim=0
    ).item()

    with _visual_in_fp32(model):
        streamed, batched = encode_both()
    diff = (streamed - batched).abs()
    max_abs, mean_abs = diff.max().item(), diff.mean().item()
    print(f"\nt={t}: fp32 max={max_abs:.3e} mean={mean_abs:.3e} | bf16 max={bf16_max:.3e} cos={cos:.6f}")
    assert max_abs <= FP32_MAX_TOL and mean_abs <= FP32_MEAN_TOL, (
        f"t={t}: batched-vs-streaming ViT features differ in fp32 "
        f"(max={max_abs:.3e} tol {FP32_MAX_TOL:.0e}, mean={mean_abs:.3e} tol {FP32_MEAN_TOL:.0e}) "
        f"— structural parity broken, not a precision issue"
    )
