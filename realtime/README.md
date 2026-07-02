# Realtime client/server speed test

Measure the streaming Qwen VLM's end-to-end speed over the real network path: a **server on the H200**
holds the model; a **client on the robot workstation** collects cam_high frames and requests inference.
Sweeps **cached** (two-frame caching) vs **full** (re-encode the 30-frame window) across early-exit
depths **8 / 16 / 36**, separating server compute from network/serialization overhead.

## Transport

Websocket + `msgpack_numpy`, identical to the OpenPI evals — the server is wire-compatible with the
robot's `openpi_client.WebsocketClientPolicy`. The server vendors `msgpack_numpy.py` (only needs
`msgpack`) and a compact `WebsocketPolicyServer` (`ws_server.py`, needs `websockets`); no `openpi`
install required. The one server-side dependency added to the qwen3vl env is `websockets`.

Per-request observation (`policy.infer(obs)`):
- `mode`: `"reset"` | `"cached"` | `"full"`
- `num_layers`: early-exit depth (1–36)
- `frames`: uint8 `[2,H,W,3]` for cached, `[30,H,W,3]` for full (already resized to send-res)
- `state`: float32 `[14]`

Response: `{ready, mode, num_layers, vision_ms, lm_ms, feature_shape, feature_norm, server_timing}`
(`ready:false` during the first 14 cached warm-up steps). `--return-features` adds the fp16 context
tensor so its network cost can be measured.

## Run

**1. Server (H200):**
```
cd /iris/projects/humanoid/qwen && PYTHONPATH=src:. \
  /iris/u/kewalk/.conda/envs/qwen3vl/bin/python realtime/server.py --host 0.0.0.0 --port 8000
```

**2a. Cluster verification — no robot** (synthetic frames over localhost, same node as the server):
```
cd /iris/projects/humanoid/qwen && PYTHONPATH=src:. \
  /iris/u/kewalk/.conda/envs/qwen3vl/bin/python realtime/dev_loopback.py --host localhost --port 8000 --steps 50
```

**2b. Real robot — LIVE (robot workstation):** opens a window showing the frame sent to the server and
prints the latency of every inference step. Needs `openpi_client` + the lerobot fork (already there).
```
python realtime/client.py --policy_host <H200-host> --port 8000 --mode cached --layers 16
```
- The window shows the newest sent cam_high frame with an overlay (`mode`, `layers`, `rtt`, `server`,
  `vis`, `lm`, `net`, and `warmup k/15` → `READY`). Press **`q`** in the window (or Ctrl-C) to stop.
- Each step also prints, e.g.:
  `step  17  rtt=44.1ms  server=41.5ms (vis=22.1 lm=17.0)  net=2.6ms  payload=662KB  22.7Hz  [READY]`
- `--mode full` streams the 30-frame window instead; `--layers {8,16,36}` sets the early-exit depth;
  `--no_viz` prints latency only (no window). The first 14 cached steps are `warmup` (cache filling).

**2c. Real robot — SWEEP (timed table over modes × layers):**
```
python realtime/client.py --policy_host <H200-host> --port 8000 --mode sweep --steps 50
```
Writes `realtime/results_realtime.md`.

## Verified loopback result (localhost, synthetic frames, H200)

End-to-end over localhost (so `net` is just serialization, not real WAN):

| mode | L | rtt_med ms | server ms | vision ms | lm ms | net ms | payload | Hz |
|---|---|---|---|---|---|---|---|---|
| cached | 8  | 48.8  | 56.7  | 33.7 | 10.0 | 3.0  | 662 KB | 16.7 |
| cached | 16 | 57.1  | 62.8  | 33.5 | 18.0 | 3.2  | 662 KB | 15.1 |
| cached | 36 | 76.4  | 82.1  | 33.4 | 38.3 | 3.1  | 662 KB | 11.7 |
| full   | 8  | 157.0 | 131.5 | 75.0 | 9.2  | 31.8 | 9.9 MB | 6.1  |
| full   | 16 | 167.5 | 139.9 | 76.3 | 17.7 | 32.4 | 5.8  |
| full   | 36 | 192.6 | 165.7 | 76.7 | 37.8 | 31.7 | 9.9 MB | 5.1  |

**Caching wins twice:** server vision 33 ms vs 75 ms (2.3×) AND payload 662 KB vs 9.9 MB (15×, so net
overhead 3 ms vs 32 ms even on localhost — the gap grows over real WAN). `lm_ms` matches the offline
experiments and scales linearly with depth. Real-robot numbers will add true network RTT on top of
`server_ms`; run 2b to fill in `results_realtime.md`.

## Notes
- The server is stateful per connection: the pair-cache accumulates across `cached` requests; send
  `{"mode":"reset"}` between phases (the clients do this automatically).
- `server_timing.infer_ms` is wall-clock for the whole `policy.infer` (includes CPU frame preprocessing),
  so it can exceed `vision_ms + lm_ms` (GPU-only). The gap is the CPU resize/patchify cost.
- Default returns no feature tensor (a 9 MB bf16/step blob would dominate latency); use
  `--return-features` to measure that path explicitly.
