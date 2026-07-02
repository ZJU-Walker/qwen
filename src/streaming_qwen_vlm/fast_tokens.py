"""FAST action tokenizer wrapper + union-vocab id math for the AR loss.

The pretrained universal FAST tokenizer (physical-intelligence/fast, DCT + BPE) turns a normalized
action chunk [horizon, action_dim] in [-1, 1] into a short discrete token sequence. Qwen's vocab is
NOT resized (tie_word_embeddings makes that invasive); instead FAST ids live in a side-car vocab
appended after Qwen's: union id = V_BASE + fast id, with a separate fast_embed / fast_head pair in
vla.py. The AR targets end with Qwen's <|im_end|>.

CLI (measures the token-length distribution so max_fast_tokens can be frozen):
    PYTHONPATH=src python -m streaming_qwen_vlm.fast_tokens --measure \
        --root ... --stats checkpoints/norm_stats.json
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import torch

V_BASE = 151936          # Qwen2.5-VL vocab_size; FAST union ids start here
IM_END_ID = 151645       # <|im_end|> — the AR stop target
PAD_INPUT_ID = 151643    # <|endoftext|> — filler for the right-padded FAST input tail
IGNORE_INDEX = -100


class FastActionTokenizer:
    """Thin wrapper around the physical-intelligence/fast AutoProcessor (needs scipy + download)."""

    def __init__(
        self,
        repo: str = "physical-intelligence/fast",
        revision: str | None = None,
        horizon: int = 30,
        action_dim: int = 14,
    ) -> None:
        from transformers import AutoProcessor  # deferred: heavy + requires trust_remote_code

        self.proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True, revision=revision)
        self.vocab_size = int(self.proc.vocab_size)
        self.horizon = horizon
        self.action_dim = action_dim

    def encode(self, chunk: np.ndarray) -> List[int]:
        """chunk float32 [horizon, action_dim] in [-1, 1] -> raw FAST ids (variable length)."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape != (self.horizon, self.action_dim):
            raise ValueError(f"expected [{self.horizon}, {self.action_dim}], got {chunk.shape}")
        ids = self.proc(chunk[None])[0]
        return [int(t) for t in ids]

    def decode(self, ids: List[int]) -> np.ndarray:
        """Raw FAST ids -> float32 [horizon, action_dim] (lossy: DCT quantization)."""
        out = self.proc.decode([list(ids)], time_horizon=self.horizon, action_dim=self.action_dim)
        return np.asarray(out[0], dtype=np.float32)

    def encode_padded(self, chunk: np.ndarray, max_tokens: int) -> Tuple[np.ndarray, int]:
        """-> (raw ids int64 [max_tokens] zero-padded, n). Raises if the chunk needs > max_tokens."""
        ids = self.encode(chunk)
        n = len(ids)
        if n > max_tokens:
            raise ValueError(
                f"FAST chunk needs {n} tokens > max_fast_tokens={max_tokens}; raise the budget "
                f"(measure with `python -m streaming_qwen_vlm.fast_tokens --measure`)"
            )
        out = np.zeros(max_tokens, dtype=np.int64)
        out[:n] = ids
        return out, n


def build_ar_row(fast_ids: np.ndarray, n: int, max_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the FAST input tail and AR targets for one sample.

    input_ids  int64 [max_tokens]:    [union(fast_0..fast_{n-1}), PAD_INPUT_ID ...]
                                      (im_end is a target only, never an input)
    targets    int64 [max_tokens+1]:  aligned to hidden positions [S_prefix-1, S_prefix+max_tokens):
                                      targets[j] = union(fast_j) for j < n; targets[n] = IM_END_ID;
                                      IGNORE_INDEX elsewhere.
    """
    if n > max_tokens:
        raise ValueError(f"n={n} > max_tokens={max_tokens}")
    input_ids = torch.full((max_tokens,), PAD_INPUT_ID, dtype=torch.long)
    input_ids[:n] = torch.as_tensor(fast_ids[:n], dtype=torch.long) + V_BASE

    targets = torch.full((max_tokens + 1,), IGNORE_INDEX, dtype=torch.long)
    targets[:n] = torch.as_tensor(fast_ids[:n], dtype=torch.long) + V_BASE
    targets[n] = IM_END_ID
    return input_ids, targets


def _measure(args: argparse.Namespace) -> None:
    """Token-length distribution over every training chunk -> pick max_fast_tokens."""
    from .normalize import load_episode_arrays, load_stats, normalize

    stats = load_stats(args.stats)
    tok = FastActionTokenizer(repo=args.repo, revision=args.revision)
    lengths = []
    for ep in stats["train_episodes"]:
        actions = load_episode_arrays(args.root, ep)["action"]
        T = len(actions)
        for t in range(10, T, args.stride):
            chunk = actions[t : t + tok.horizon]
            if len(chunk) < tok.horizon:  # hold-pose padding, same as the dataloader
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], tok.horizon - len(chunk), 0)])
            lengths.append(len(tok.encode(normalize(chunk, stats["action"]))))
    arr = np.asarray(lengths)
    print(f"chunks={len(arr)}  min={arr.min()}  p50={int(np.median(arr))}  "
          f"p95={int(np.percentile(arr, 95))}  p99={int(np.percentile(arr, 99))}  max={arr.max()}")
    print(f"suggested max_fast_tokens >= {arr.max()} (current config default 128)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--root", default="/iris/projects/humanoid/trossen_data/0528_merge_block_mem")
    ap.add_argument("--stats", default="checkpoints/norm_stats.json")
    ap.add_argument("--repo", default="physical-intelligence/fast")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--stride", type=int, default=5, help="control-step stride when measuring")
    args = ap.parse_args()
    if args.measure:
        _measure(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
