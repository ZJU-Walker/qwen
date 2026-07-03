#!/usr/bin/env python
"""Offline replay: run an ActPolicy checkpoint over one episode and compare predicted vs demo joints.

Simulates deployment ticks over the pre-extracted cam_high JPGs: at tick k the pair
(frame 20k, frame 20k+10) + state[20k+10] goes to the policy; the returned [30,14] chunk is
compared against the demonstrated actions a[t : t+30]. Writes <out>/replay_ep{N}.npz (predicted
chunks, demo actions, per-tick MSE) and, if matplotlib is importable, per-joint curve PNGs.

    cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
      python examples/replay_policy.py --checkpoint checkpoints/overfit/step_002000 --episode 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streaming_qwen_vlm.normalize import load_episode_arrays  # noqa: E402
from streaming_qwen_vlm.policy import ActPolicy  # noqa: E402
from streaming_qwen_vlm.training.dataset import frame_stride, tick_stride  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="step_XXXXXX checkpoint dir")
    ap.add_argument("--root", default="/iris/projects/humanoid/trossen_data/0528_merge_block_mem")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default: <checkpoint>/replay)")
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.checkpoint, "replay")
    os.makedirs(out_dir, exist_ok=True)

    policy = ActPolicy(args.checkpoint, use_ema=not args.no_ema)
    arrs = load_episode_arrays(args.root, args.episode)
    actions, states = arrs["action"], arrs["state"]
    frames_dir = os.path.join(args.root, "frames", "cam_high", f"episode_{args.episode:06d}")

    def frame(idx: int) -> np.ndarray:
        return np.asarray(
            Image.open(os.path.join(frames_dir, f"frame_{idx:06d}.jpg")).convert("RGB"), dtype=np.uint8
        )

    T = len(actions)
    horizon = policy.expert_cfg.horizon
    # Tick grid derives from the CHECKPOINT's config (meta.json) — run1-style (10/20) and
    # run3-style (15/30) checkpoints both replay correctly with this one script.
    f_stride = frame_stride(policy.cfg.fps)
    t_stride = tick_stride(policy.cfg)
    preds, targets, ticks, mses = [], [], [], []
    policy.reset()
    for k in range((T - f_stride) // t_stride + 1):
        t = t_stride * k + f_stride
        if t >= T:
            break
        pair = np.stack([frame(t_stride * k), frame(t)])
        out = policy.act(pair, states[t])
        demo = actions[t : t + horizon]
        if len(demo) < horizon:
            demo = np.concatenate([demo, np.repeat(demo[-1:], horizon - len(demo), axis=0)])
        mse = float(((out["actions"] - demo) ** 2).mean())
        tm = out["timings"]
        print(f"tick {k:3d} (t={t:3d})  mse={mse:.6f}  "
              f"vis={tm['vision_ms']:.0f}ms prefill={tm['prefill_ms']:.0f}ms "
              f"denoise={tm['denoise_ms']:.0f}ms")
        preds.append(out["actions"])
        targets.append(demo)
        ticks.append(t)
        mses.append(mse)

    preds, targets = np.stack(preds), np.stack(targets)
    npz = os.path.join(out_dir, f"replay_ep{args.episode}.npz")
    np.savez(npz, preds=preds, targets=targets, ticks=np.asarray(ticks), mse=np.asarray(mses))
    print(f"\nmean open-loop MSE over {len(ticks)} ticks: {np.mean(mses):.6f}\nwrote {npz}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Stitch tick-aligned chunk starts into trajectories per joint (first 20 actions per tick).
        fig, axes = plt.subplots(7, 2, figsize=(14, 18), sharex=True)
        for j in range(14):
            ax = axes[j % 7][j // 7]
            ax.plot(np.arange(T), actions[:, j], "k-", lw=1, label="demo")
            for k, t in enumerate(ticks):
                seg = preds[k][: max(0, min(t_stride, T - t)), j]
                ax.plot(np.arange(t, t + len(seg)), seg, lw=1, alpha=0.8)
            ax.set_title(f"joint {j}", fontsize=8)
        axes[0][0].legend()
        png = os.path.join(out_dir, f"replay_ep{args.episode}.png")
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib not installed; skipped plots (npz has everything)")


if __name__ == "__main__":
    main()
