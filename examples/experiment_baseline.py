"""BASELINE (no caching): re-encode the WHOLE 30-frame window every step.

Contrast with examples/experiment_layers.py (two-frame caching). Here each step runs all 30 frames
through the vision tower as one [15,24,24] get_video_features call (2160 vision tokens) instead of
encoding only the single new pair (144 tokens). The LLM early-exit forward is identical, so the
difference is purely the vision cost the cache eliminates.

Same clean methodology: backbone loaded once, CPU preprocessing done once (GPU-only timed region),
global pre-warm, warm-up discarded, mean/median over many timed steps. Depths 8 / 16 / 36.

Run:
    cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
      /iris/u/kewalk/.conda/envs/qwen3vl/bin/python examples/experiment_baseline.py \
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
from streaming_qwen_vlm.early_exit import run_language_early_exit  # noqa: E402
from streaming_qwen_vlm.preprocess import frames_to_video_inputs, prepare_frames  # noqa: E402
from streaming_qwen_vlm.timing import cuda_timer, peak_memory, reset_peak_memory  # noqa: E402
from run_episode import load_episode  # noqa: E402

LAYERS = [8, 16, 36]
WARMUP_STEPS = 10
TIMED_STEPS = 50


@torch.inference_mode()
def gpu_update_baseline(vlm, model, full_pv, full_grid, state_tok, num_layers, norm):
    """No-cache update: encode the FULL 30-frame window + inject + LLM forward at depth N."""
    with cuda_timer() as t_vis:
        video_embeds = model.get_video_features(full_pv, full_grid)[0].to(vlm.device, vlm.dtype)
    with cuda_timer() as t_lm:
        inputs_embeds = model.get_input_embeddings()(vlm._input_ids)
        inputs_embeds = inputs_embeds.masked_scatter(
            vlm._video_mask.unsqueeze(-1), video_embeds.reshape(-1)
        )
        inputs_embeds = torch.cat([inputs_embeds, state_tok], dim=1)
        run_language_early_exit(
            model, inputs_embeds, vlm._attn_full, vlm._position_ids, num_layers, norm
        )
    return t_vis(), t_lm()


def run_depth(model, processor, num_layers, frames, states, full_pv, full_grid):
    cfg = VLMConfig(num_llm_layers_to_run=num_layers)
    vlm = StreamingQwenVLM(cfg, model=model, processor=processor)
    state_tok = vlm.state_proj(torch.as_tensor(states[-1]).to(vlm.device)).to(vlm.device, vlm.dtype)

    for _ in range(WARMUP_STEPS):
        gpu_update_baseline(vlm, model, full_pv, full_grid, state_tok, num_layers, cfg.early_exit_norm)
    torch.cuda.synchronize()

    reset_peak_memory()
    vis_ms, lm_ms, tot_ms = [], [], []
    for _ in range(TIMED_STEPS):
        v, l = gpu_update_baseline(vlm, model, full_pv, full_grid, state_tok, num_layers, cfg.early_exit_norm)
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

    # Pre-process the full 30-frame window ONCE (CPU work out of the timed region).
    vinp = frames_to_video_inputs(processor, prepare_frames(frames, cfg0), cfg0)
    full_pv = vinp["pixel_values_videos"].to(model.device)
    full_grid = vinp["video_grid_thw"].to(model.device)

    print("Global GPU pre-warm ...")
    run_depth(model, processor, 36, frames, states, full_pv, full_grid)

    rows = [run_depth(model, processor, n, frames, states, full_pv, full_grid) for n in LAYERS]

    print(f"\nBASELINE (no cache, full-window encode), mean over {TIMED_STEPS} steps, episode {args.episode}:\n")
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
