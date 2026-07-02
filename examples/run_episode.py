"""Stream a real Trossen cam_high episode through StreamingQwenVLM.

Reads the pre-extracted cam_high JPG frames + the 14-dim observation.state from the LeRobot-format
dataset, subsamples a 30-frame window, and feeds it pair-by-pair to the streaming VLM. Output is
available from the first step (front-padded window, D6); prints per-step latency and the final
context-token output.

Example:
    cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
      /iris/u/kewalk/.conda/envs/qwen3vl/bin/python examples/run_episode.py \
        --root /iris/projects/humanoid/trossen_data/0528_merge_block_mem \
        --episode 0 --layers 16

Notes:
- cam_high frames are 960x540 (16:9); VLMConfig pins a SQUARE 336x336 grid, so prepare_frames squashes
  the aspect ratio. That matches the current tests/benchmark. Change cfg.fixed_resolution for 16:9.
- A 30-frame @ 3 fps window is 10 s, but episodes here are only ~6-7 s. To fill the window from one
  episode we EVENLY subsample `window_frames` frames across the episode; the effective sampling rate
  therefore differs from cfg.fps (the mrope temporal positions still use cfg.fps via the fixed
  template). Pass --stride to sample at a fixed native stride instead (may yield < window_frames).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streaming_qwen_vlm import StreamingQwenVLM, VLMConfig  # noqa: E402
from streaming_qwen_vlm.outputs import (  # noqa: E402
    TOKEN_TYPE_STATE,
    TOKEN_TYPE_TEXT,
    TOKEN_TYPE_VIDEO,
)

CAM = "cam_high"


def _episode_paths(root: str, episode: int):
    frames_dir = os.path.join(root, "frames", CAM, f"episode_{episode:06d}")
    parquet = os.path.join(root, "data", "chunk-000", f"episode_{episode:06d}.parquet")
    return frames_dir, parquet


def load_episode(root: str, episode: int, window_frames: int, stride: int | None):
    frames_dir, parquet = _episode_paths(root, episode)
    jpgs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    n = len(jpgs)
    if n == 0:
        raise RuntimeError(f"No JPG frames in {frames_dir}")

    if stride is not None:
        idx = list(range(0, n, stride))[:window_frames]
        if len(idx) < window_frames:
            print(
                f"[warn] stride={stride} on a {n}-frame episode yields only {len(idx)} frames "
                f"(< window_frames={window_frames}); the window will not fill."
            )
    else:
        # Evenly sample exactly window_frames frames across the whole episode.
        idx = np.linspace(0, n - 1, window_frames).round().astype(int).tolist()

    states = pq.read_table(parquet, columns=["observation.state"]).column("observation.state").to_pylist()

    frames = []
    sel_states = []
    for i in idx:
        im = Image.open(os.path.join(frames_dir, jpgs[i])).convert("RGB")
        frames.append(np.asarray(im, dtype=np.uint8))
        si = min(i, len(states) - 1)
        sel_states.append(np.asarray(states[si], dtype=np.float32))
    return frames, sel_states, idx, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/iris/projects/humanoid/trossen_data/0528_merge_block_mem")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--layers", type=int, default=16, help="num_llm_layers_to_run (early-exit depth)")
    ap.add_argument("--stride", type=int, default=None, help="fixed native-frame stride (default: even-sample)")
    ap.add_argument("--instruction", default=None)
    args = ap.parse_args()

    cfg_kwargs = dict(num_llm_layers_to_run=args.layers)
    if args.instruction:
        cfg_kwargs["instruction"] = args.instruction
    cfg = VLMConfig(**cfg_kwargs)

    frames, states, idx, n_total = load_episode(args.root, args.episode, cfg.window_frames, args.stride)
    print(
        f"Episode {args.episode}: {n_total} native cam_high frames -> sampled {len(frames)} "
        f"(native size {frames[0].shape[1]}x{frames[0].shape[0]} WxH -> model {cfg.fixed_resolution})"
    )

    print("Loading Qwen2.5-VL-3B backbone ...")
    vlm = StreamingQwenVLM(cfg)

    out = None
    n_pairs = len(frames) // cfg.frames_per_pair
    for i in range(n_pairs):
        a, b = frames[2 * i], frames[2 * i + 1]
        state = states[2 * i + 1]  # newest frame's state for this step
        t0 = time.perf_counter()
        out = vlm.step([a, b], state)
        dt = (time.perf_counter() - t0) * 1000
        # Front-padding (D6): output at every step; early steps use a window padded with pair 0.
        status = f"OUTPUT (pairs {min(i + 1, cfg.num_pairs)}/{cfg.num_pairs})"
        print(f"  step {i + 1:2d}/{n_pairs}  pair-frames=({idx[2*i]},{idx[2*i+1]})  {dt:6.1f} ms  -> {status}")

    if out is None:
        print("\nNo pairs were fed (episode too short?).")
        return

    tt = out.token_types[0]
    print("\n=== Final VLMOutput ===")
    print(f"  context_tokens : {tuple(out.context_tokens.shape)}  dtype={out.context_tokens.dtype}")
    print(f"  context_mask   : {tuple(out.context_mask.shape)}")
    print(f"  token_types    : {tuple(out.token_types.shape)}")
    print(f"  layers run     : {out.layer_index}")
    print(
        f"  token counts   : video={int((tt == TOKEN_TYPE_VIDEO).sum())} "
        f"text={int((tt == TOKEN_TYPE_TEXT).sum())} state={int((tt == TOKEN_TYPE_STATE).sum())}"
    )
    print(f"  metadata       : {out.metadata}")


if __name__ == "__main__":
    main()
