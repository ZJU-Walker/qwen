"""Per-dim quantile normalization for actions and robot state (pi0-style q01/q99 -> [-1, 1]).

The Trossen dataset stores no dataset-wide stats (only per-episode min/max/mean/std), so quantiles
are computed here from the raw parquet files over the TRAIN split only, then frozen to a JSON file
that both training and deployment load.

Degenerate dims: the left arm is static and the left gripper is completely dead in this dataset
(range 0.0), so any per-dim scaling must guard against a near-zero q01..q99 range. Guarded dims
normalize to 0.0, discretize to the middle bin, and unnormalize back to q01.

CLI:
    PYTHONPATH=src python -m streaming_qwen_vlm.normalize \
        --root /iris/projects/humanoid/trossen_data/0528_merge_block_mem --out checkpoints/norm_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq

DEGENERATE_EPS = 1e-6


@dataclass
class NormStats:
    q01: np.ndarray  # float32 [dim]
    q99: np.ndarray  # float32 [dim]

    @property
    def range(self) -> np.ndarray:
        return self.q99 - self.q01

    @property
    def degenerate(self) -> np.ndarray:
        """Bool [dim]; True where the q01..q99 range is too small to scale by."""
        return self.range < DEGENERATE_EPS


def split_episodes(num_episodes: int = 61, holdout_every: int = 10) -> Tuple[List[int], List[int]]:
    """Deterministic train/val split: every ``holdout_every``-th episode (offset -1) is validation."""
    val = [i for i in range(num_episodes) if i % holdout_every == holdout_every - 1]
    train = [i for i in range(num_episodes) if i not in val]
    return train, val


def _episode_parquet(root: str, episode: int) -> str:
    return os.path.join(root, "data", "chunk-000", f"episode_{episode:06d}.parquet")


def load_episode_arrays(root: str, episode: int) -> Dict[str, np.ndarray]:
    """Return {'action': [T,14], 'state': [T,14]} float32 for one episode."""
    table = pq.read_table(_episode_parquet(root, episode), columns=["action", "observation.state"])
    action = np.stack(table.column("action").to_pylist()).astype(np.float32)
    state = np.stack(table.column("observation.state").to_pylist()).astype(np.float32)
    return {"action": action, "state": state}


def compute_norm_stats(root: str, episodes: List[int]) -> Dict[str, NormStats]:
    actions, states = [], []
    for ep in episodes:
        arrs = load_episode_arrays(root, ep)
        actions.append(arrs["action"])
        states.append(arrs["state"])
    a = np.concatenate(actions, axis=0)
    s = np.concatenate(states, axis=0)
    return {
        "action": NormStats(
            q01=np.quantile(a, 0.01, axis=0).astype(np.float32),
            q99=np.quantile(a, 0.99, axis=0).astype(np.float32),
        ),
        "state": NormStats(
            q01=np.quantile(s, 0.01, axis=0).astype(np.float32),
            q99=np.quantile(s, 0.99, axis=0).astype(np.float32),
        ),
    }


def normalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """[.., dim] raw -> [-1, 1] (unclipped, pi0 convention); degenerate dims -> 0.0."""
    x = np.asarray(x, dtype=np.float32)
    safe_range = np.where(stats.degenerate, 1.0, stats.range)
    out = 2.0 * (x - stats.q01) / safe_range - 1.0
    return np.where(stats.degenerate, 0.0, out).astype(np.float32)


def unnormalize(xn: np.ndarray, stats: NormStats) -> np.ndarray:
    """[-1, 1] -> raw units; degenerate dims collapse to q01 (== the constant raw value)."""
    xn = np.asarray(xn, dtype=np.float32)
    out = (xn + 1.0) / 2.0 * stats.range + stats.q01
    return np.where(stats.degenerate, stats.q01, out).astype(np.float32)


def discretize(x: np.ndarray, stats: NormStats, bins: int = 256) -> np.ndarray:
    """Raw [dim] -> int64 bin indices in [0, bins-1]; degenerate dims land in the middle bin."""
    n = np.clip(normalize(x, stats), -1.0, 1.0)
    return np.round((n + 1.0) / 2.0 * (bins - 1)).astype(np.int64)


def save_stats(path: str, stats: Dict[str, NormStats], train_eps: List[int], val_eps: List[int]) -> None:
    payload = {
        "train_episodes": train_eps,
        "val_episodes": val_eps,
        **{
            key: {"q01": s.q01.tolist(), "q99": s.q99.tolist()}
            for key, s in stats.items()
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_stats(path: str) -> Dict[str, object]:
    """Return {'action': NormStats, 'state': NormStats, 'train_episodes': [...], 'val_episodes': [...]}."""
    with open(path) as f:
        payload = json.load(f)
    out: Dict[str, object] = {
        "train_episodes": payload["train_episodes"],
        "val_episodes": payload["val_episodes"],
    }
    for key in ("action", "state"):
        out[key] = NormStats(
            q01=np.asarray(payload[key]["q01"], dtype=np.float32),
            q99=np.asarray(payload[key]["q99"], dtype=np.float32),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/iris/projects/humanoid/trossen_data/0528_merge_block_mem")
    ap.add_argument("--out", default="checkpoints/norm_stats.json")
    ap.add_argument("--num-episodes", type=int, default=61)
    ap.add_argument("--holdout-every", type=int, default=10)
    args = ap.parse_args()

    train_eps, val_eps = split_episodes(args.num_episodes, args.holdout_every)
    stats = compute_norm_stats(args.root, train_eps)

    # Sanity: the right arm (dims 7..13) must be non-degenerate; the left arm is expected static.
    act = stats["action"]
    right = ~act.degenerate[7:14]
    if not right.any():
        raise RuntimeError(f"All right-arm action dims degenerate — wrong dataset? ranges={act.range}")
    for name, s in stats.items():
        print(f"{name}: q01={np.round(s.q01, 4).tolist()}")
        print(f"{name}: q99={np.round(s.q99, 4).tolist()}")
        print(f"{name}: degenerate dims={np.nonzero(s.degenerate)[0].tolist()}")

    save_stats(args.out, stats, train_eps, val_eps)
    print(f"wrote {args.out}  (train={len(train_eps)} eps, val={val_eps})")


if __name__ == "__main__":
    main()
