"""Rolling cache of per-pair vision embeddings.

Each new 2-frame pair is encoded ONCE via get_video_features and stored as a [tokens_per_pair, 2048]
tensor. concat() reconstructs the full [total_video_tokens, 2048] window feature in chronological
order. This is the core of the two-frame caching trick: the vision encoder runs on 2 frames per step,
never on the whole 30-frame window.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import torch


class PairVisionCache:
    def __init__(self, num_pairs: int, tokens_per_pair: int, hidden: int = 2048) -> None:
        self.num_pairs = num_pairs
        self.tokens_per_pair = tokens_per_pair
        self.hidden = hidden
        self._buf: Deque[torch.Tensor] = deque(maxlen=num_pairs)

    @torch.inference_mode()
    def encode_pair(self, model, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode a single pair -> [tokens_per_pair, 2048]. grid_thw must be [[1, grid_h, grid_w]]."""
        dev = model.device
        feats = model.get_video_features(pixel_values.to(dev), grid_thw.to(dev))
        # get_video_features returns a tuple split per video; one pair -> one element.
        z = feats[0]
        if z.shape[0] != self.tokens_per_pair or z.shape[-1] != self.hidden:
            raise ValueError(
                f"Encoded pair has shape {tuple(z.shape)}, expected "
                f"[{self.tokens_per_pair}, {self.hidden}]"
            )
        return z

    def push(self, z: torch.Tensor) -> None:
        """Append a pair embedding; the deque auto-evicts the oldest when full."""
        self._buf.append(z)

    def concat(self, pad_to_full: bool = False) -> torch.Tensor:
        """Chronological [total_video_tokens, 2048] (oldest pair first).

        With ``pad_to_full`` (Stage-2 D6 front-padding), a not-yet-full cache is left-padded by
        repeating the OLDEST pair so output is available from the very first tick:
        ``[pair_0 x (num_pairs - len), pair_0..pair_k]``. The training dataloader applies the
        identical padding so streaming and training see bit-identical windows.
        """
        if not self._buf:
            raise RuntimeError("PairVisionCache is empty; nothing to concat.")
        pairs = list(self._buf)
        if pad_to_full and len(pairs) < self.num_pairs:
            pairs = [pairs[0]] * (self.num_pairs - len(pairs)) + pairs
        return torch.cat(pairs, dim=0)

    def is_full(self) -> bool:
        return len(self._buf) == self.num_pairs

    def __len__(self) -> int:
        return len(self._buf)
