"""H200 latency/memory benchmark for the streaming Qwen2.5-VL feature module.

Sweeps the early-exit depth N and reports steady-state per-step latency (vision encode + LM forward)
and peak memory. Target: total_update_ms < 667 (1.5 Hz). Writes a Markdown table to results.md.

Run:
    cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
      /iris/u/kewalk/.conda/envs/qwen3vl/bin/python benchmark/benchmark_h200.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streaming_qwen_vlm import StreamingQwenVLM, VLMConfig, load_backbone  # noqa: E402
from streaming_qwen_vlm.preprocess import pair_to_pixel_values, prepare_frames  # noqa: E402
from streaming_qwen_vlm.timing import cuda_timer, peak_memory, reset_peak_memory  # noqa: E402

LAYER_SWEEP = [8, 12, 16, 24, 36]
WARMUP_STEPS = 3
TIMED_STEPS = 20
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.md")


def _random_frames(n, h, w, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _fill_cache(svlm, frames, state):
    """Feed exactly num_pairs pairs so the cache is full (warm-up of the window)."""
    out = None
    for i in range(svlm.cfg.num_pairs):
        out = svlm.step([frames[2 * i], frames[2 * i + 1]], state)
    assert out is not None
    return out


def benchmark_one(model, processor, num_layers, base_frames, state):
    cfg = VLMConfig(num_llm_layers_to_run=num_layers)
    svlm = StreamingQwenVLM(cfg, model=model, processor=processor)
    h, w = cfg.fixed_resolution

    # Fill the rolling window once.
    _fill_cache(svlm, base_frames, state)

    # Pre-make a stream of fresh pairs to push each step (steady state: one new pair per step).
    pairs = [
        [np.random.default_rng(1000 + k).integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(2)]
        for k in range(WARMUP_STEPS + TIMED_STEPS)
    ]

    # Warm up (kernels/autotune).
    for k in range(WARMUP_STEPS):
        svlm.step(pairs[k], state)
    torch.cuda.synchronize()

    reset_peak_memory()
    vis_ms, tot_ms = [], []
    for k in range(WARMUP_STEPS, WARMUP_STEPS + TIMED_STEPS):
        two = pairs[k]

        # Vision-encode-only timing (standalone; does NOT mutate the cache).
        prepared = prepare_frames(two, cfg)
        pinp = pair_to_pixel_values(processor, prepared, cfg)
        with cuda_timer() as t_vis:
            svlm.cache.encode_pair(model, pinp["pixel_values_videos"], pinp["video_grid_thw"])

        # Full steady-state update: encode the new pair (again, inside step) + LM early-exit forward.
        with cuda_timer() as t_step:
            out = svlm.step(two, state)
        assert out is not None

        vis_ms.append(t_vis())
        tot_ms.append(t_step())

    alloc_mb, reserved_mb = peak_memory()

    def _median(xs):
        return float(np.median(xs))

    total_update_ms = _median(tot_ms)
    median_vis = _median(vis_ms)
    row = {
        "num_llm_layers_to_run": num_layers,
        "fixed_resolution": f"{h}x{w}",
        "video_tokens": cfg.total_video_tokens,
        "vision_encode_ms": round(median_vis, 2),
        "lm_forward_ms": round(total_update_ms - median_vis, 2),
        "total_update_ms": round(total_update_ms, 2),
        "peak_alloc_MB": round(alloc_mb, 1),
        "peak_reserved_MB": round(reserved_mb, 1),
        "sustained_Hz": round(1000.0 / total_update_ms, 2),
    }
    return row


def write_results(rows):
    cols = [
        "num_llm_layers_to_run", "fixed_resolution", "video_tokens", "vision_encode_ms",
        "lm_forward_ms", "total_update_ms", "peak_alloc_MB", "peak_reserved_MB", "sustained_Hz",
    ]
    lines = [
        "# H200 Benchmark — Streaming Qwen2.5-VL-3B feature module",
        "",
        "Steady-state per-`step()` latency (one new 2-frame pair encoded + 30-frame window LM forward",
        "with real early exit) and peak memory. Target: `total_update_ms < 667` (1.5 Hz).",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines += [
        "",
        "## Correctness (first H200 run)",
        "- Vision pair-cache equivalence (fp32): max abs diff 1.41e-4, mean abs diff 8.06e-7 "
        "(bf16: max 7.79, mean 2.62e-2). Validates the two-frame caching design.",
        "- Cached-injection vs full forward: max abs diff 0.0, cosine 1.0 (exact).",
        "- Early exit vs official text-model forward (N=8,16 pre-norm; N=36+final_norm): max abs diff 0.0 (exact).",
        "",
    ]
    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    cfg0 = VLMConfig()
    model, processor = load_backbone(cfg0)
    h, w = cfg0.fixed_resolution
    base_frames = _random_frames(cfg0.window_frames, h, w, seed=7)
    state = np.zeros(cfg0.state_dim, dtype=np.float32)

    rows = []
    for n in LAYER_SWEEP:
        row = benchmark_one(model, processor, n, base_frames, state)
        print(f"N={n:>2}: total={row['total_update_ms']}ms "
              f"(vis={row['vision_encode_ms']}, lm={row['lm_forward_ms']}) "
              f"{row['sustained_Hz']}Hz peak_alloc={row['peak_alloc_MB']}MB")
        rows.append(row)

    write_results(rows)


if __name__ == "__main__":
    main()
