"""Fixed-resolution frame preparation -> Qwen video pixel values + grid_thw.

We always go through the processor's video path (never hand-implement patchification). The grid is
pinned by passing pre-resized frames AND min_pixels==max_pixels==H*W so smart_resize is a no-op.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from .config import VLMConfig


def prepare_frames(frames: Sequence[np.ndarray], cfg: VLMConfig) -> np.ndarray:
    """Resize each RGB frame to cfg.fixed_resolution and stack -> uint8 [T, H, W, 3].

    Accepts frames as HxWx3 uint8 arrays (any input size). Bicubic resize.
    """
    h, w = cfg.fixed_resolution
    out: List[np.ndarray] = []
    for f in frames:
        arr = np.asarray(f)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Each frame must be HxWx3 RGB, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[0] != h or arr.shape[1] != w:
            arr = np.asarray(Image.fromarray(arr).resize((w, h), Image.BICUBIC))
        out.append(arr)
    return np.stack(out, axis=0)


def _video_inputs(processor, frames_thwc: np.ndarray, cfg: VLMConfig) -> Dict[str, torch.Tensor]:
    """Run the processor video path on one [T,H,W,3] clip; pin the grid via min==max pixels."""
    h, w = cfg.fixed_resolution
    vp = getattr(processor, "video_processor", None)
    if vp is None:
        raise RuntimeError("processor has no video_processor; cannot run the video path.")
    out = vp(
        videos=[frames_thwc],
        min_pixels=h * w,
        max_pixels=h * w,
        return_tensors="pt",
    )
    n_videos = len(out["video_grid_thw"])
    out["second_per_grid_ts"] = [cfg.second_per_grid_ts] * n_videos
    return out


def frames_to_video_inputs(processor, frames_thwc: np.ndarray, cfg: VLMConfig) -> Dict[str, torch.Tensor]:
    """Full rolling window ([window_frames, H, W, 3]) -> one logical video row.

    Returns pixel_values_videos, video_grid_thw == [[num_pairs, grid_h, grid_w]], second_per_grid_ts.
    """
    if frames_thwc.shape[0] != cfg.window_frames:
        raise ValueError(
            f"Expected {cfg.window_frames} frames for the full window, got {frames_thwc.shape[0]}"
        )
    out = _video_inputs(processor, frames_thwc, cfg)
    return out


def pair_to_pixel_values(processor, two_frames_thwc: np.ndarray, cfg: VLMConfig) -> Dict[str, torch.Tensor]:
    """One pair ([2, H, W, 3]) -> video_grid_thw == [[1, grid_h, grid_w]], tokens_per_pair tokens."""
    if two_frames_thwc.shape[0] != cfg.frames_per_pair:
        raise ValueError(
            f"Expected {cfg.frames_per_pair} frames for a pair, got {two_frames_thwc.shape[0]}"
        )
    return _video_inputs(processor, two_frames_thwc, cfg)
