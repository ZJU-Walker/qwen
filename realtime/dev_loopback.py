"""Dev loopback client to verify the server on the cluster (NO robot needed).

Sends synthetic frames over the same websocket/msgpack_numpy protocol the robot uses, sweeping
{cached, full} x {8,16,36} layers, and prints the latency table. Runs in the qwen3vl env (uses the
vendored msgpack_numpy + websockets.sync), so you can validate the protocol + both modes + all depths
end-to-end against a localhost server before touching the robot.

Run (with the server already up on the same node):
    cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
      /iris/u/kewalk/.conda/envs/qwen3vl/bin/python realtime/dev_loopback.py --host localhost --port 8000
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
from websockets.sync.client import connect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtime import msgpack_numpy  # noqa: E402

MODES = ["cached", "full"]
LAYERS = [8, 16, 36]
WINDOW_FRAMES = 30
SEND_RES = 336


class WSClient:
    """Minimal msgpack_numpy websocket client (wire-compatible with the robot's openpi_client)."""

    def __init__(self, host: str, port: int):
        uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._ws = connect(uri, max_size=None, compression=None)
        self._packer = msgpack_numpy.Packer()
        self.metadata = msgpack_numpy.unpackb(self._ws.recv())

    def infer(self, obs: dict):
        data = self._packer.pack(obs)
        self._ws.send(data)
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"server error:\n{resp}")
        return msgpack_numpy.unpackb(resp), len(data)

    def close(self):
        self._ws.close()


def _frames(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, SEND_RES, SEND_RES, 3), dtype=np.uint8)


def run_phase(client, mode, layers, steps, warmup):
    state = np.zeros(14, dtype=np.float32)
    n_send = 2 if mode == "cached" else WINDOW_FRAMES

    client.infer({"mode": "reset"})
    # Warm up (cached: fill the 15-pair cache; both: warm kernels). Discarded.
    for k in range(warmup):
        client.infer({"mode": mode, "num_layers": layers, "frames": _frames(n_send, k), "state": state})

    rtt, infer_ms, vis_ms, lm_ms, payload = [], [], [], [], 0
    for k in range(steps):
        obs = {"mode": mode, "num_layers": layers, "frames": _frames(n_send, 1000 + k), "state": state}
        t0 = time.perf_counter()
        resp, payload = client.infer(obs)
        rtt.append((time.perf_counter() - t0) * 1000.0)
        infer_ms.append(resp["server_timing"]["infer_ms"])
        vis_ms.append(resp.get("vision_ms", 0.0))
        lm_ms.append(resp.get("lm_ms", 0.0))
    return {
        "mode": mode, "layers": layers,
        "rtt_mean": statistics.mean(rtt), "rtt_med": statistics.median(rtt),
        "rtt_p95": np.percentile(rtt, 95),
        "server_ms": statistics.mean(infer_ms),
        "vision_ms": statistics.mean(vis_ms), "lm_ms": statistics.mean(lm_ms),
        "net_ms": statistics.mean(rtt) - statistics.mean(infer_ms),
        "payload_KB": payload / 1024.0, "hz": 1000.0 / statistics.mean(rtt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=18)
    args = ap.parse_args()

    client = WSClient(args.host, args.port)
    print("connected; server metadata:", client.metadata)

    rows = []
    for mode in MODES:
        for layers in LAYERS:
            r = run_phase(client, mode, layers, args.steps, args.warmup)
            print(f"  {mode:>6} L{layers:<2}: rtt={r['rtt_med']:.1f}ms server={r['server_ms']:.1f} "
                  f"(vis={r['vision_ms']:.1f} lm={r['lm_ms']:.1f}) net={r['net_ms']:.1f} "
                  f"payload={r['payload_KB']:.0f}KB {r['hz']:.1f}Hz")
            rows.append(r)
    client.close()

    hdr = (f"\n{'mode':>6} {'L':>3} | {'rtt_med':>7} {'rtt_p95':>7} | {'server':>6} {'vis':>5} {'lm':>5} "
           f"| {'net':>5} | {'payload':>8} | {'Hz':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['mode']:>6} {r['layers']:>3} | {r['rtt_med']:>7.1f} {r['rtt_p95']:>7.1f} | "
              f"{r['server_ms']:>6.1f} {r['vision_ms']:>5.1f} {r['lm_ms']:>5.1f} | {r['net_ms']:>5.1f} | "
              f"{r['payload_KB']:>7.0f}K | {r['hz']:>5.1f}")


if __name__ == "__main__":
    main()
