"""T5: front-padding semantics (D6) — CPU-only, no model.

The streaming cache and the training dataloader must build bit-identical windows: a not-yet-full
window is left-padded by repeating the OLDEST pair, i.e. [pair_0 x (15-k), pairs 0..k].
"""

import pytest
import torch

from streaming_qwen_vlm.training.dataset import MIN_T, pair_frame_indices
from streaming_qwen_vlm.vision_cache import PairVisionCache

NUM_PAIRS = 15


def _tagged(k: int) -> torch.Tensor:
    return torch.full((4, 8), float(k))


def test_cache_front_padding_order():
    cache = PairVisionCache(NUM_PAIRS, tokens_per_pair=4, hidden=8)
    for k in range(5):  # push pairs 0..4 only
        cache.push(_tagged(k))
    out = cache.concat(pad_to_full=True).view(NUM_PAIRS, 4, 8)
    expected = [0] * (NUM_PAIRS - 5) + [0, 1, 2, 3, 4]
    assert [int(out[i, 0, 0]) for i in range(NUM_PAIRS)] == expected


def test_cache_full_is_unaffected():
    cache = PairVisionCache(NUM_PAIRS, tokens_per_pair=4, hidden=8)
    for k in range(NUM_PAIRS + 3):  # overflow: oldest evicted
        cache.push(_tagged(k))
    padded = cache.concat(pad_to_full=True)
    plain = cache.concat()
    assert torch.equal(padded, plain)
    assert int(padded.view(NUM_PAIRS, 4, 8)[0, 0, 0]) == 3  # pairs 3..17 remain


def test_cache_empty_raises():
    cache = PairVisionCache(NUM_PAIRS, tokens_per_pair=4, hidden=8)
    with pytest.raises(RuntimeError):
        cache.concat(pad_to_full=True)


def test_pair_frame_indices_matches_cache_padding():
    """The dataloader's index-level padding equals the cache's tensor-level padding."""
    for t in (MIN_T, 29, 30, 49, 130, 291):
        ks = pair_frame_indices(t, NUM_PAIRS)
        assert len(ks) == NUM_PAIRS
        k_max = (t - MIN_T) // 20
        # simulate streaming: pairs 0..k_max pushed in order, then front-padded
        cache = PairVisionCache(NUM_PAIRS, tokens_per_pair=1, hidden=1)
        for k in range(k_max + 1):
            cache.push(torch.tensor([[float(k)]]))
        streamed = cache.concat(pad_to_full=True).view(-1).tolist()
        expected = [float(min(k, k_max)) for k in ks]  # overflow: cache keeps latest 15
        if k_max >= NUM_PAIRS:
            expected = [float(k) for k in range(k_max - NUM_PAIRS + 1, k_max + 1)]
            assert ks[-1] == k_max and ks == list(range(k_max - NUM_PAIRS + 1, k_max + 1))
        assert streamed == expected


def test_pair_frame_indices_rejects_early_t():
    with pytest.raises(ValueError):
        pair_frame_indices(MIN_T - 1, NUM_PAIRS)


def test_collate_dedup_offsets():
    """collate must offset each sample's slot_map into the batch-global unique-pair rows."""
    from streaming_qwen_vlm.config import VLMConfig
    from streaming_qwen_vlm.training.dataset import collate

    cfg = VLMConfig()
    rows = cfg.grid_h * cfg.grid_w

    def item(n_unique: int, tag: float, slot_map):
        # pixel rows of unique pair u carry value 100*tag + u, so gathers are traceable
        pv = (torch.arange(n_unique).repeat_interleave(rows).float() + 100.0 * tag)
        return {
            "pixel_values": pv.unsqueeze(-1).expand(n_unique * rows, 4).clone(),
            "slot_map": torch.tensor(slot_map, dtype=torch.long),
            "state_ids": torch.zeros(56, dtype=torch.long),
            "fast_input_ids": torch.zeros(8, dtype=torch.long),
            "ar_targets": torch.zeros(9, dtype=torch.long),
            "actions_norm": torch.zeros(30, 14),
            "episode": torch.tensor(0),
            "t": torch.tensor(0),
        }

    a = item(2, 1, [0] * 14 + [1])
    b = item(3, 2, [0] * 13 + [1, 2])
    out = collate([a, b], cfg)

    assert out["pixel_values"].shape == (5 * rows, 4)
    assert out["video_grid_thw"].shape == (5, 3)
    assert (out["video_grid_thw"] == torch.tensor(cfg.pair_grid_thw)).all()
    # sample b's local indices offset by sample a's 2 unique pairs
    assert out["slot_map"][0].tolist() == [0] * 14 + [1]
    assert out["slot_map"][1].tolist() == [2] * 13 + [3, 4]
    # gathering per-pair rows through the global slot_map reproduces each sample's window
    pair_vals = out["pixel_values"].view(5, rows, 4)[:, 0, 0]  # one value per unique pair
    assert pair_vals[out["slot_map"][0]].tolist() == [100.0] * 14 + [101.0]
    assert pair_vals[out["slot_map"][1]].tolist() == [200.0] * 13 + [201.0, 202.0]
