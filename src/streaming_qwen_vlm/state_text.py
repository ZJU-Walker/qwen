"""Render discretized robot state as constant-length text token ids.

Each of the 14 state dims becomes " %03d" (bin 0..255): one space token + three single-digit
tokens = 4 ids per dim, 56 ids total. Qwen's tokenizer splits every digit into its own token, so
the count is constant for ANY bin values — which makes S_prefix a project-wide constant (static
masks, positions, expert key layout, realtime template). ``build_state_lut`` asserts this on the
live tokenizer at startup rather than trusting it.

No per-sample tokenizer calls: rendering is a LUT gather.
"""

from __future__ import annotations

import torch

NUM_BINS = 256
IDS_PER_DIM = 4  # [space, d0, d1, d2]
STATE_DIMS = 14
N_STATE_TOKENS = STATE_DIMS * IDS_PER_DIM  # 56


def build_state_lut(tokenizer) -> torch.LongTensor:
    """[256, 4] LongTensor: token ids of " %03d" for every bin value.

    Asserts the constant-length property: " " is exactly 1 id and every 3-digit string is exactly
    3 ids, and that composing per-piece ids equals tokenizing the full rendered string (i.e. no
    cross-piece BPE merges).
    """
    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    if len(space_ids) != 1:
        raise AssertionError(f'tokenizer encodes " " to {space_ids}, expected exactly 1 id')

    lut = torch.zeros(NUM_BINS, IDS_PER_DIM, dtype=torch.long)
    for b in range(NUM_BINS):
        digit_ids = tokenizer.encode(f"{b:03d}", add_special_tokens=False)
        if len(digit_ids) != 3:
            raise AssertionError(
                f'tokenizer encodes "{b:03d}" to {len(digit_ids)} ids ({digit_ids}), expected 3 '
                f"single-digit ids — constant-length state text does not hold for this tokenizer"
            )
        lut[b] = torch.tensor(space_ids + digit_ids, dtype=torch.long)

    # Cross-piece merge check on a few representative bin vectors.
    for bins in ([0] * STATE_DIMS, [255] * STATE_DIMS, list(range(100, 100 + STATE_DIMS)), [128] * STATE_DIMS):
        rendered = "".join(f" {b:03d}" for b in bins)
        full = tokenizer.encode(rendered, add_special_tokens=False)
        composed = lut[torch.tensor(bins)].reshape(-1).tolist()
        if full != composed:
            raise AssertionError(
                f"full-string tokenization differs from LUT composition for bins={bins}:\n"
                f"  full={full}\n  composed={composed}"
            )
    return lut


def bins_to_ids(bins, lut: torch.LongTensor) -> torch.LongTensor:
    """int bins [14] (list/np/tensor) -> token ids [56]."""
    bins_t = torch.as_tensor(bins, dtype=torch.long)
    if bins_t.shape != (STATE_DIMS,):
        raise ValueError(f"expected [{STATE_DIMS}] bins, got {tuple(bins_t.shape)}")
    return lut[bins_t].reshape(-1)
