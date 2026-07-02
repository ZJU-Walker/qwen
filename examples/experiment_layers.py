"""Average steady-state GPU step latency for early-exit depths 8 / 16 / 36 (last layer).

Methodology (to get clean, comparable numbers):
- Backbone loaded ONCE; the rolling window is pre-filled from a real cam_high episode.
- All CPU image preprocessing (PIL resize + processor) is done ONCE, before timing — so the timed
  region is GPU-only: encode one new pair (vision) + LLM early-exit forward at depth N.
- A global GPU pre-warm runs before any measurement so the first depth isn't penalized by clock ramp.
- Per depth: WARMUP_STEPS discarded, then TIMED_STEPS timed; report mean, median and std.

Run:
    cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
      /iris/u/kewalk/.conda/envs/qwen3vl/bin/python examples/experiment_layers.py \
        --root /iris/projects/humanoid/trossen_data/0528_merge_block_mem --episode 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streaming_qwen_vlm import StreamingQwenVLM, VLMConfig, load_backbone  # noqa: E402
from streaming_qwen_vlm.config import VIDEO_TOKEN_ID  # noqa: E402
from streaming_qwen_vlm.early_exit import run_language_early_exit  # noqa: E402
from streaming_qwen_vlm.preprocess import pair_to_pixel_values, prepare_frames  # noqa: E402
from streaming_qwen_vlm.timing import cuda_timer, peak_memory, reset_peak_memory  # noqa: E402
from run_episode import load_episode  # noqa: E402

LAYERS = [8, 16, 36]
WARMUP_STEPS = 10
TIMED_STEPS = 50


@torch.inference_mode()
def gpu_update(vlm, model, pv, grid, state_tok, num_layers, norm):
    """The GPU-only steady-state update: encode new pair + inject + LLM forward at depth N.

    Mirrors StreamingQwenVLM.step but takes a PRE-PROCESSED pair (pv, grid) and a PRE-PROJECTED state
    token, so no CPU preprocessing pollutes the timed region. Returns (vision_ms, lm_ms).
    """
    with cuda_timer() as t_vis:
        z = vlm.cache.encode_pair(model, pv, grid)
        vlm.cache.push(z.to(vlm.dtype))
    with cuda_timer() as t_lm:
        video_embeds = vlm.cache.concat().to(vlm.device, vlm.dtype)
        inputs_embeds = model.get_input_embeddings()(vlm._input_ids)
        inputs_embeds = inputs_embeds.masked_scatter(
            vlm._video_mask.unsqueeze(-1), video_embeds.reshape(-1)
        )
        inputs_embeds = torch.cat([inputs_embeds, state_tok], dim=1)
        run_language_early_exit(
            model, inputs_embeds, vlm._attn_full, vlm._position_ids, num_layers, norm
        )
    return t_vis(), t_lm()


def run_depth(model, processor, num_layers, frames, states, pv_pool, grid):
    cfg = VLMConfig(num_llm_layers_to_run=num_layers)
    vlm = StreamingQwenVLM(cfg, model=model, processor=processor)

    # Fill the window once.
    out = None
    for i in range(cfg.num_pairs):
        out = vlm.step([frames[2 * i], frames[2 * i + 1]], states[2 * i + 1])
    assert out is not None

    state_tok = vlm.state_proj(
        torch.as_tensor(states[-1]).to(vlm.device)
    ).to(vlm.device, vlm.dtype)

    n = len(pv_pool)
    for k in range(WARMUP_STEPS):
        gpu_update(vlm, model, pv_pool[k % n], grid, state_tok, num_layers, cfg.early_exit_norm)
    torch.cuda.synchronize()

    reset_peak_memory()
    vis_ms, lm_ms, tot_ms = [], [], []
    for k in range(TIMED_STEPS):
        v, l = gpu_update(vlm, model, pv_pool[k % n], grid, state_tok, num_layers, cfg.early_exit_norm)
        vis_ms.append(v)
        lm_ms.append(l)
        tot_ms.append(v + l)
    alloc_mb, _ = peak_memory()

    tot = np.array(tot_ms)
    return {
        "layers": num_layers,
        "avg_total_ms": float(tot.mean()),
        "median_total_ms": float(np.median(tot)),
        "std_total_ms": float(tot.std()),
        "avg_vision_ms": float(np.mean(vis_ms)),
        "avg_lm_ms": float(np.mean(lm_ms)),
        "avg_hz": 1000.0 / float(tot.mean()),
        "peak_alloc_MB": alloc_mb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/iris/projects/humanoid/trossen_data/0528_merge_block_mem")
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()

    cfg0 = VLMConfig()
    frames, states, _, _ = load_episode(args.root, args.episode, cfg0.window_frames, None)

    print("Loading Qwen2.5-VL-3B backbone once ...")
    model, processor = load_backbone(cfg0)

    # Pre-process a pool of fresh pairs ONCE (CPU work out of the timed region).
    pv_pool, grid = [], None
    for k in range(max(WARMUP_STEPS, TIMED_STEPS)):
        two = [frames[(2 * k) % len(frames)], frames[(2 * k + 1) % len(frames)]]
        pinp = pair_to_pixel_values(processor, prepare_frames(two, cfg0), cfg0)
        pv_pool.append(pinp["pixel_values_videos"].to(model.device))
        grid = pinp["video_grid_thw"].to(model.device)

    # Global GPU pre-warm so the first measured depth isn't penalized by clock ramp.
    print("Global GPU pre-warm ...")
    run_depth(model, processor, 36, frames, states, pv_pool, grid)

    rows = [run_depth(model, processor, n, frames, states, pv_pool, grid) for n in LAYERS]

    print(f"\nGPU-only steady-state, mean over {TIMED_STEPS} steps (warm-up discarded), episode {args.episode}:\n")
    hdr = (f"{'layers':>6} | {'avg_ms':>7} | {'median_ms':>9} | {'±std':>6} | "
           f"{'vision_ms':>9} | {'lm_ms':>7} | {'avg_Hz':>6} | {'peak_MB':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['layers']:>6} | {r['avg_total_ms']:>7.2f} | {r['median_total_ms']:>9.2f} | "
            f"{r['std_total_ms']:>6.2f} | {r['avg_vision_ms']:>9.2f} | {r['avg_lm_ms']:>7.2f} | "
            f"{r['avg_hz']:>6.2f} | {r['peak_alloc_MB']:>8.1f}"
        )


if __name__ == "__main__":
    main()
