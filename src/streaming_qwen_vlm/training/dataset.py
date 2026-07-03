"""Tick-aligned, front-padded training dataset over the Trossen LeRobot episodes.

Sampling (claude_plan §3 + D5/D6): any control step t >= 10 is a sample (~10k samples across 61
episodes, a ~20x multiplier over tick-only sampling). The visual window is built on the deployment
tick grid — dataset 30 fps, VLM 3 fps => frame stride 10, pair k = frames (20k, 20k+10) — using the
pairs fully observed by t, front-padded by repeating the OLDEST pair exactly like
``PairVisionCache.concat(pad_to_full=True)`` (bit-identical windows, gated by
tests/test_dataloader_streaming_parity.py). Visual staleness <= 19 control steps matches
deployment's within-chunk staleness by construction.

Targets: actions a[t : t+horizon] at native 30 Hz, padded past the episode end by repeating the
last action (hold-pose; deliberate deviation from loss-masking so the FAST tokenizer sees a full
chunk). The same normalized chunk feeds the flow target and the FAST/AR targets.

Frames go through the exact ``prepare_frames`` + ``pair_to_pixel_values`` calls the streaming
server uses. The FAST tokenizer is lazily constructed per dataloader worker (trust_remote_code
objects are not reliably picklable).

ViT dedup: ``pixel_values`` holds each sample's UNIQUE pairs only, plus a ``slot_map`` [num_pairs]
that expands their embeddings into the 15 window slots (vla.forward_train gathers). Episodes here
(<= 209 frames) never contain more than 10 unique pairs at any t (mean ~5), so encoding all 15
slots would re-run the ViT on front-pad duplicates ~3x for nothing. The gather is mathematically
identical to encoding duplicates: gradients through repeated slots sum into the single encode by
linearity.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import VLMConfig
from ..fast_tokens import FastActionTokenizer, build_ar_row
from ..normalize import NormStats, discretize, load_episode_arrays, normalize
from ..preprocess import pair_to_pixel_values, prepare_frames
from ..state_text import bins_to_ids

FRAME_STRIDE = 10   # 30 fps -> 3 fps
TICK_STRIDE = 20    # control steps per completed pair (= per deployment tick)
MIN_T = 10          # first control step with a complete pair (frames 0, 10)


def pair_frame_indices(t: int, num_pairs: int) -> List[int]:
    """The <= num_pairs pair indices (front-padded with 0) whose frames are all <= t."""
    if t < MIN_T:
        raise ValueError(f"t={t} < {MIN_T}: no complete pair exists yet")
    k_max = (t - MIN_T) // TICK_STRIDE
    return [max(0, k) for k in range(k_max - num_pairs + 1, k_max + 1)]


class TrossenActDataset(Dataset):
    def __init__(
        self,
        root: str,
        episodes: Sequence[int],
        cfg: VLMConfig,
        processor,                       # Qwen AutoProcessor (shared; fork-inherited by workers)
        state_lut: torch.Tensor,         # [256, 4] from build_state_lut / ActTemplate.state_lut
        action_stats: NormStats,
        state_stats: NormStats,
        max_fast_tokens: int,
        horizon: int = 30,
        fast_repo: str = "physical-intelligence/fast",
        fast_revision: Optional[str] = None,
        tick_aligned: bool = False,      # True: only deployment-tick t values (for eval)
    ) -> None:
        self.root = root
        self.cfg = cfg
        self.processor = processor
        self.state_lut = state_lut
        self.action_stats = action_stats
        self.state_stats = state_stats
        self.max_fast_tokens = max_fast_tokens
        self.horizon = horizon
        self._fast_args = dict(repo=fast_repo, revision=fast_revision,
                               horizon=horizon, action_dim=cfg.state_dim)
        self._fast: Optional[FastActionTokenizer] = None  # lazy per-worker

        # Preload all low-dim arrays (~1.2 MB total) and build the sample index.
        self.arrays: Dict[int, Dict[str, np.ndarray]] = {}
        self.samples: List[tuple] = []
        for ep in episodes:
            arrs = load_episode_arrays(root, ep)
            self.arrays[ep] = arrs
            T = len(arrs["action"])
            n_jpgs = len([f for f in os.listdir(self._frames_dir(ep)) if f.endswith(".jpg")])
            if n_jpgs != T:
                raise RuntimeError(f"episode {ep}: {n_jpgs} JPGs != {T} parquet rows")
            step = TICK_STRIDE if tick_aligned else 1
            self.samples.extend((ep, t) for t in range(MIN_T, T, step))

    # --- helpers ---
    def _frames_dir(self, ep: int) -> str:
        return os.path.join(self.root, "frames", "cam_high", f"episode_{ep:06d}")

    def _load_frame(self, ep: int, idx: int) -> np.ndarray:
        path = os.path.join(self._frames_dir(ep), f"frame_{idx:06d}.jpg")
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    @property
    def fast(self) -> FastActionTokenizer:
        if self._fast is None:
            self._fast = FastActionTokenizer(**self._fast_args)
        return self._fast

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ep, t = self.samples[i]
        arrs = self.arrays[ep]
        cfg = self.cfg

        # 1. Visual window: UNIQUE pairs only (chronological); slot_map expands them into the
        #    15 window slots downstream (front-pad = repeated slot 0, never a repeated encode).
        ks = pair_frame_indices(t, cfg.num_pairs)
        unique_ks = list(dict.fromkeys(ks))  # order-preserving == chronological
        pv_by_k: Dict[int, torch.Tensor] = {}
        for k in unique_ks:
            fa = self._load_frame(ep, TICK_STRIDE * k)
            fb = self._load_frame(ep, TICK_STRIDE * k + FRAME_STRIDE)
            prepared = prepare_frames([fa, fb], cfg)
            pv_by_k[k] = pair_to_pixel_values(self.processor, prepared, cfg)["pixel_values_videos"]
        pixel_values = torch.cat([pv_by_k[k] for k in unique_ks], dim=0)  # [n_unique*576, 1176] f32
        slot_map = torch.tensor([unique_ks.index(k) for k in ks], dtype=torch.long)  # [num_pairs]

        # 2. State -> bins -> constant-length token ids.
        bins = discretize(arrs["state"][t], self.state_stats, bins=cfg.state_bins)
        state_ids = bins_to_ids(bins, self.state_lut)

        # 3. Action chunk, hold-pose padded, normalized; shared by flow and FAST targets.
        chunk = arrs["action"][t : t + self.horizon]
        if len(chunk) < self.horizon:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], self.horizon - len(chunk), axis=0)], axis=0
            )
        actions_norm = normalize(chunk, self.action_stats)

        fast_ids, n = self.fast.encode_padded(actions_norm, self.max_fast_tokens)
        fast_input_ids, ar_targets = build_ar_row(fast_ids, n, self.max_fast_tokens)

        return {
            "pixel_values": pixel_values,
            "slot_map": slot_map,
            "state_ids": state_ids,
            "fast_input_ids": fast_input_ids,
            "ar_targets": ar_targets,
            "actions_norm": torch.from_numpy(actions_norm),
            "episode": torch.tensor(ep),
            "t": torch.tensor(t),
        }


def collate(batch: List[Dict[str, torch.Tensor]], cfg: VLMConfig) -> Dict[str, torch.Tensor]:
    B = len(batch)
    # Variable unique-pair counts per sample: cat the unique pixel rows, and offset each sample's
    # slot_map so it indexes into the batch-global list of encoded pairs.
    rows_per_pair = cfg.grid_h * cfg.grid_w  # pre-merge patch rows per pair (576 at 336x336)
    n_unique = [b["pixel_values"].shape[0] // rows_per_pair for b in batch]
    offsets = torch.tensor([0] + n_unique[:-1], dtype=torch.long).cumsum(0)
    grid = torch.tensor(cfg.pair_grid_thw, dtype=torch.long).unsqueeze(0).repeat(sum(n_unique), 1)
    return {
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),  # [N_u*576, 1176]
        "video_grid_thw": grid,                                                # [N_u, 3]
        "slot_map": torch.stack([b["slot_map"] + off                           # [B, P] -> rows of
                                 for b, off in zip(batch, offsets)]),          #   the N_u encodes
        "state_ids": torch.stack([b["state_ids"] for b in batch]),             # [B, 56]
        "fast_input_ids": torch.stack([b["fast_input_ids"] for b in batch]),   # [B, T_fast]
        "ar_targets": torch.stack([b["ar_targets"] for b in batch]),           # [B, T_fast+1]
        "actions_norm": torch.stack([b["actions_norm"] for b in batch]),       # [B, 30, 14]
        "episode": torch.stack([b["episode"] for b in batch]),
        "t": torch.stack([b["t"] for b in batch]),
    }
