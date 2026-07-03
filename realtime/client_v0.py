"""Robot-workstation client: stream cam_high to the Qwen server and measure realtime speed.

Runs on the robot computer (has `openpi_client` + the lerobot fork). Captures cam_high frames, streams
them to the H200 server, and sweeps {cached, full} x {8,16,36} early-exit depths, reporting end-to-end
round-trip latency vs server inference time vs network overhead. Writes realtime/results_realtime.md.

Usage (robot workstation):
    python realtime/client.py --policy_host <H200-host> --port 8000 --steps 50

Mirrors the capture pattern of examples/trossen_ai/eval_real_hierarchical.py (RobotSource): cam_high is
captured HWC uint8 and resized to the model resolution before sending (small payload).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque

import numpy as np

# ========================= EDIT THESE FOR THE ROBOT RUN ========================= #
# Set once here and run `python realtime/client.py` with NO flags (act mode). CLI
# flags still exist and override these if you pass them.
SERVER_IP = "iris-hgx-1"          # the GPU box running realtime/server.py --checkpoint ...
SERVER_PORT = 8000
LEROBOT_FORK_PATH = "/home/iris/lerobot"   # path to the lerobot Trossen fork on THIS robot machine
EXECUTE = False                   # False = dry run (prints actions, no motion); True = actuate
VIZ = True                        # show the 16-frame window (big current pair + 4x4 grid), threaded
GRAVITY_COMP_TIME = 5.0           # seconds to hand-place both arms before actuation (execute only)
MAX_ACTION_DELTA = 0.15           # max abs joint move per control step (rad) — rate limit
# ================================================================================ #

MODES = ["cached", "full"]
LAYERS = [8, 16, 36]
WINDOW_FRAMES = 30
RESULTS_PATH = "realtime/results_realtime.md"


# --------------------------------------------------------------------------- #
class RobotCamera:
    """cam_high capture via the lerobot Trossen fork (lazy import)."""

    def __init__(self, send_res: int, lerobot_path: str = LEROBOT_FORK_PATH):
        if lerobot_path not in sys.path:
            sys.path.insert(0, lerobot_path)
        from lerobot.common.robot_devices.robots.configs import TrossenAISoloRobotConfig
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config

        cfg = TrossenAISoloRobotConfig(
            max_relative_target=None, min_time_to_move_multiplier=4.0, camera_interface="opencv"
        )
        self.robot = make_robot_from_config(cfg)
        self.robot.connect()
        self.send_res = send_res
        import cv2  # noqa
        self._cv2 = cv2

    def capture(self):
        """Return (frame HWC uint8 [send_res,send_res,3], state float32 [14])."""
        obs = self.robot.capture_observation()
        state = obs["observation.state"].detach().cpu().numpy().astype(np.float32)
        img = obs["observation.images.cam_high"].detach().cpu().numpy()  # HWC uint8
        img = self._cv2.resize(img, (self.send_res, self.send_res))
        return img.astype(np.uint8), state

    def act(self, action: np.ndarray):
        """Send one absolute joint-target action [14] to the robot."""
        import torch

        self.robot.send_action(torch.from_numpy(np.asarray(action, dtype=np.float32)))

    def gravity_comp_warmup(self, duration: float = 5.0):
        """Torque-off the follower arms (gravity-comp mode) so the operator can hand-place them,
        then latch + lock the chosen pose. Ports openpi's eval_real_RTC_working_warmup.py:
        Torque_Enable=0 = external-effort/gravity-comp on the Trossen fork (arm is back-drivable
        but holds against gravity); on lock we re-enable torque at the current Present_Position.
        Returns the locked [14] pose so the caller can seed rate-limiting from it.
        """
        import time

        if duration <= 0:
            return None
        for arm in self.robot.follower_arms.values():
            arm.write("Torque_Enable", 0)
        end = time.monotonic() + duration
        last = None
        print(f"GRAVITY-COMP: move the arms to the desired start pose ({duration:.0f}s)...")
        while time.monotonic() < end:
            rem = max(0, int(np.ceil(end - time.monotonic())))
            if rem != last:
                print(f"  choose initial pose: {rem}s remaining", flush=True)
                last = rem
            time.sleep(0.05)
        locked = []
        for name, arm in self.robot.follower_arms.items():
            pos = arm.read("Present_Position").astype(np.float32)
            arm.write("Torque_Enable", 1)
            arm.write("Goal_Position", pos)
            locked.append(pos)
            print(f"  locked {name} at {np.round(pos, 4).tolist()}")
        return np.concatenate(locked).astype(np.float32)

    def close(self):
        self.robot.disconnect()


# --------------------------------------------------------------------------- #
def run_phase(client, cam, mode, layers, steps, warmup, history):
    """One sweep cell. cached: send 2 newest frames; full: send the 30-frame window."""
    n_send = 2 if mode == "cached" else WINDOW_FRAMES
    client.infer({"mode": "reset"})

    def grab_obs():
        img, state = cam.capture()
        history.append(img)
        while len(history) < n_send:           # bootstrap: duplicate until enough history
            history.append(img)
        frames = np.stack(list(history)[-n_send:], axis=0)
        return {"mode": mode, "num_layers": layers, "frames": frames, "state": state}

    for _ in range(warmup):
        client.infer(grab_obs())

    rtt, server_ms, vis_ms, lm_ms, payload = [], [], [], [], 0
    for _ in range(steps):
        obs = grab_obs()
        payload = len(client._packer.pack(obs)) if hasattr(client, "_packer") else 0
        t0 = time.perf_counter()
        resp = client.infer(obs)
        rtt.append((time.perf_counter() - t0) * 1000.0)
        server_ms.append(resp["server_timing"]["infer_ms"])
        vis_ms.append(resp.get("vision_ms", 0.0))
        lm_ms.append(resp.get("lm_ms", 0.0))

    return {
        "mode": mode, "layers": layers,
        "rtt_med": statistics.median(rtt), "rtt_p95": float(np.percentile(rtt, 95)),
        "server_ms": statistics.mean(server_ms),
        "vision_ms": statistics.mean(vis_ms), "lm_ms": statistics.mean(lm_ms),
        "net_ms": statistics.median(rtt) - statistics.mean(server_ms),
        "payload_KB": payload / 1024.0, "hz": 1000.0 / statistics.median(rtt),
    }


def write_results(rows, meta):
    lines = [
        "# Realtime client/server speed test (robot cam_high -> H200 Qwen server)",
        "",
        f"Server: {meta}",
        "",
        "Per-step end-to-end: `rtt` = client round-trip (send->infer->recv); `server` = server-side",
        "infer wall time; `net` = rtt - server (network + serialization); `payload` = request bytes.",
        "",
        "| mode | layers | rtt_med_ms | rtt_p95_ms | server_ms | vision_ms | lm_ms | net_ms | payload_KB | Hz |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['layers']} | {r['rtt_med']:.1f} | {r['rtt_p95']:.1f} | "
            f"{r['server_ms']:.1f} | {r['vision_ms']:.1f} | {r['lm_ms']:.1f} | {r['net_ms']:.1f} | "
            f"{r['payload_KB']:.0f} | {r['hz']:.1f} |"
        )
    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {RESULTS_PATH}")


def run_live(client, cam, mode, layers, send_res, no_viz=False):
    """Continuous loop: capture -> send -> show the sent frame + print latency every step. Press 'q' to quit."""
    cv2 = None
    if not no_viz:
        import cv2  # only needed for the window

    n_send = 2 if mode == "cached" else WINDOW_FRAMES
    history = deque(maxlen=WINDOW_FRAMES)
    win = f"qwen sent frame [{mode} L{layers}]"
    client.infer({"mode": "reset"})
    print(f"LIVE {mode} L{layers}: streaming cam_high to the server. Press 'q' in the window (or Ctrl-C) to stop.")

    step = 0
    try:
        while True:
            img, state = cam.capture()          # img: RGB [send_res,send_res,3]
            history.append(img)
            while len(history) < n_send:
                history.append(img)
            frames = np.stack(list(history)[-n_send:], axis=0)

            obs = {"mode": mode, "num_layers": layers, "frames": frames, "state": state}
            payload_kb = len(client._packer.pack(obs)) / 1024.0
            t0 = time.perf_counter()
            resp = client.infer(obs)
            rtt = (time.perf_counter() - t0) * 1000.0

            srv = resp["server_timing"]["infer_ms"]
            vis, lm = resp.get("vision_ms", 0.0), resp.get("lm_ms", 0.0)
            ready = resp.get("ready", True)
            tag = "READY" if ready else f"warmup {resp.get('num_pairs', '?')}/15"
            print(f"step {step:4d}  rtt={rtt:6.1f}ms  server={srv:6.1f}ms (vis={vis:5.1f} lm={lm:5.1f})  "
                  f"net={rtt - srv:5.1f}ms  payload={payload_kb:.0f}KB  {1000.0 / rtt:5.1f}Hz  [{tag}]")

            if not no_viz:
                disp = cv2.cvtColor(history[-1], cv2.COLOR_RGB2BGR)   # newest sent frame
                disp = cv2.resize(disp, (512, 512), interpolation=cv2.INTER_NEAREST)
                lines = [f"{mode}  L{layers}  [{tag}]",
                         f"rtt {rtt:.0f}ms  server {srv:.0f}ms",
                         f"vis {vis:.0f}  lm {lm:.0f}  net {rtt - srv:.0f}ms"]
                y = 24
                for txt in lines:
                    cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                    y += 26
                cv2.imshow(win, disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            step += 1
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if not no_viz:
            cv2.destroyAllWindows()


class WindowViz:
    """Background-threaded mirror of the server's rolling window.

    The CONTROL LOOP only calls push()/update_overlay() — lock-protected numpy writes, ~zero cost,
    and it NEVER touches cv2. A daemon thread owns ALL OpenCV calls (GUI calls are not thread-safe)
    and redraws the big-current-pair + 4x4-grid canvas at ~15 fps from the latest snapshot, polling
    waitKey for 'q'. Rendering is decoupled from the control loop, so a slow imshow (e.g. over X11
    forwarding) can never stall control. Check should_stop() for the 'q' keypress.
    """

    def __init__(self, window_frames: int, cols: int = 4, big: int = 384, cell: int = 160,
                 fps: float = 15.0):
        import threading

        self.window_frames = window_frames
        self.cols = cols
        self.rows = (window_frames + cols - 1) // cols
        self.big, self.cell = big, cell
        self.win = "qwen window (sent to server)"
        self._period = 1.0 / fps
        self._lock = threading.Lock()
        self._frames = deque(maxlen=window_frames)  # RGB uint8, oldest..newest
        self._overlay = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # --- called from the control loop (cheap, no cv2) ---
    def push(self, frame_a, frame_b):
        with self._lock:
            self._frames.append(np.asarray(frame_a, dtype=np.uint8))
            self._frames.append(np.asarray(frame_b, dtype=np.uint8))

    def update_overlay(self, lines):
        with self._lock:
            self._overlay = list(lines)

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    # --- render thread: owns every cv2 call ---
    def _run(self):
        import time as _time

        import cv2

        while not self._stop.is_set():
            t0 = _time.perf_counter()
            with self._lock:
                frames = list(self._frames)
                overlay = list(self._overlay)
            if frames:
                cv2.imshow(self.win, self._build(cv2, frames, overlay))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self._stop.set()
                    break
            _time.sleep(max(0.0, self._period - (_time.perf_counter() - t0)))
        cv2.destroyAllWindows()

    def _build(self, cv2, frames, overlay):
        while len(frames) < self.window_frames:  # front-pad like the server (repeat oldest)
            frames.insert(0, frames[0])

        def bgr(f, size):
            return cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (size, size),
                              interpolation=cv2.INTER_NEAREST)

        top = np.hstack([bgr(frames[-2], self.big), bgr(frames[-1], self.big)])
        cv2.putText(top, "current pair (sent this tick)", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cells = []
        for i, f in enumerate(frames):
            c = bgr(f, self.cell)
            if i >= self.window_frames - 2:  # newest pair
                cv2.rectangle(c, (1, 1), (self.cell - 2, self.cell - 2), (0, 255, 0), 3)
            cv2.putText(c, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cells.append(c)
        rows = [np.hstack(cells[r * self.cols:(r + 1) * self.cols]) for r in range(self.rows)]
        grid = np.vstack(rows)

        W = max(top.shape[1], grid.shape[1])
        def padw(img):
            if img.shape[1] < W:
                img = np.hstack([img, np.zeros((img.shape[0], W - img.shape[1], 3), np.uint8)])
            return img
        canvas = np.vstack([padw(top), padw(grid)])

        for k, line in enumerate(overlay):
            y = 58 + k * 26
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        return canvas


def run_act(client, cam, execute: bool, meta: dict | None = None,
            gravity_comp_time: float = 5.0, max_action_delta: float = 0.15, viz: bool = False):
    """Stage-2 VLA loop against a --checkpoint server (mode 'act').

    The capture/execution cadence derives from the SERVER's metadata (fps, frames_per_pair,
    control_hz), so one client serves any checkpoint config: run1-style (3 fps -> tick every 20
    control steps) or run3-style (2 fps -> tick every 30). Each tick sends the 2 newest frames
    (captured 1/fps s apart, matching the training pair grid) + current state in a background
    thread; the previous chunk keeps executing meanwhile, and the ``min(idx, len-1)`` clamp holds
    the last action through the fetch gap.

    Safety (ported from openpi eval_real_RTC_working_warmup.py): with --execute, a gravity-comp
    warmup lets the operator hand-place BOTH follower arms (incl. the static left arm) at the start
    pose before actuation; then every commanded action is rate-limited so absolute-position targets
    can't jump more than ``max_action_delta`` rad/control-step. Without --execute, actions are
    printed, not sent (and the warmup is skipped).
    """
    import threading

    meta = meta or {}
    control_hz = float(meta.get("control_hz", 30))
    fps = int(meta.get("fps", 3))
    frames_per_pair = int(meta.get("frames_per_pair", 2))
    frame_gap = int(round(control_hz / fps))          # control steps between the pair's 2 frames
    exec_per_tick = frames_per_pair * frame_gap       # a new pair completes every tick

    period = 1.0 / control_hz
    window_frames = int(meta.get("window_frames", 16))
    viewer = WindowViz(window_frames) if viz else None
    # Warmup FIRST (arms back-drivable), then reset the server cache and start streaming.
    last_action = cam.gravity_comp_warmup(gravity_comp_time) if execute else None
    client.infer({"mode": "reset"})
    print(f"ACT loop: tick every {exec_per_tick} steps @ {control_hz:.0f} Hz (fps={fps}) "
          f"({'EXECUTING' if execute else 'dry run — pass --execute to actuate'})"
          f"{' +viz' if viz else ''}. Ctrl-C{' or q in window' if viz else ''} to stop.")

    result = {}  # cross-thread: {"resp": dict, "t_sent": float}
    lock = threading.Lock()
    last_meta = {"rtt": 0.0, "vis": 0.0, "prefill": 0.0, "denoise": 0.0, "pairs": "?"}

    def request(frames, state, t_sent):
        t0 = time.perf_counter()
        resp = client.infer({"mode": "act", "frames": frames, "state": state})
        with lock:
            result["resp"], result["t_sent"], result["rtt"] = resp, t_sent, (time.perf_counter() - t0) * 1e3

    chunk, idx, frame_a, tick = None, 0, None, 0
    try:
        i = 0
        while True:
            t_iter = time.perf_counter()
            if i % exec_per_tick == exec_per_tick - frame_gap:  # capture the pair's first frame
                frame_a, _ = cam.capture()
            if i % exec_per_tick == 0:                        # tick: capture second frame + state, fire request
                frame_b, state = cam.capture()
                fa = frame_a if frame_a is not None else frame_b
                if viewer is not None:
                    viewer.push(fa, frame_b)                  # mirror the sent pair into the window
                threading.Thread(target=request, daemon=True,
                                 args=(np.stack([fa, frame_b]), state, time.perf_counter())).start()
                tick += 1
            with lock:
                resp = result.pop("resp", None)
                if resp is not None:
                    late = int(round((time.perf_counter() - result.pop("t_sent")) * control_hz))
                    chunk, idx = np.asarray(resp["actions"]), min(late, 9)
                    n_pairs = int(meta.get("window_frames", 30)) // frames_per_pair
                    last_meta = {"rtt": result.pop("rtt"), "vis": resp["vision_ms"],
                                 "prefill": resp["prefill_ms"], "denoise": resp["denoise_ms"],
                                 "pairs": resp.get("num_pairs", "?")}
                    print(f"tick {tick:4d}  rtt={last_meta['rtt']:6.1f}ms  "
                          f"(vis={resp['vision_ms']:.0f} prefill={resp['prefill_ms']:.0f} "
                          f"denoise={resp['denoise_ms']:.0f})  start_idx={idx}  "
                          f"pairs={resp.get('num_pairs', '?')}/{n_pairs}")
            if chunk is not None:
                action = chunk[min(idx, len(chunk) - 1)]
                idx += 1
                if execute:
                    # Rate-limit: clip the per-control-step move so a bad absolute target can't jump.
                    if last_action is not None:
                        delta = np.clip(action - last_action, -max_action_delta, max_action_delta)
                        action = (last_action + delta).astype(np.float32)
                    last_action = action
                    cam.act(action)
                elif i % exec_per_tick == 0:
                    print(f"    action[0:4]={np.round(action[:4], 3).tolist()} ...")
            if viewer is not None:                            # cheap: overlay text + q check
                viewer.update_overlay([
                    f"tick {tick}  {control_hz:.0f}Hz  {'EXEC' if execute else 'DRY'}",
                    f"rtt {last_meta['rtt']:.0f}ms  vis {last_meta['vis']:.0f}  "
                    f"prefill {last_meta['prefill']:.0f}  denoise {last_meta['denoise']:.0f}  "
                    f"pairs {last_meta['pairs']}"])
                if viewer.should_stop():
                    break
            i += 1
            time.sleep(max(0.0, period - (time.perf_counter() - t_iter)))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if viewer is not None:
            viewer.close()


def main():
    # Defaults come from the CONFIG BLOCK at the top of this file, so `python realtime/client.py`
    # runs act mode with no flags. Flags still override when passed.
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_host", default=SERVER_IP, help="server host (default: SERVER_IP const)")
    ap.add_argument("--port", type=int, default=SERVER_PORT)
    ap.add_argument("--mode", default="act", choices=["cached", "full", "sweep", "act"],
                    help="act (default) = VLA action execution; cached/full = feature loop; sweep = table")
    ap.add_argument("--execute", action="store_true", default=EXECUTE,
                    help="act mode: actually send actions (default from EXECUTE const; dry-run prints)")
    ap.add_argument("--layers", type=int, default=16, help="early-exit depth for live mode")
    ap.add_argument("--no_viz", action="store_true", help="live mode: print latency only, no window")
    ap.add_argument("--steps", type=int, default=50, help="sweep mode: timed steps per cell")
    ap.add_argument("--warmup", type=int, default=18, help=">=15 fills the cached pair-cache")
    ap.add_argument("--send_res", type=int, default=336,
                    help="capture resolution (overridden by the server's checkpoint resolution)")
    ap.add_argument("--lerobot-path", default=LEROBOT_FORK_PATH,
                    help="path to the lerobot Trossen fork on this robot machine")
    ap.add_argument("--gravity-comp-time", type=float, default=GRAVITY_COMP_TIME,
                    help="act+execute: seconds of gravity-comp warmup to hand-place the arms")
    ap.add_argument("--max-action-delta", type=float, default=MAX_ACTION_DELTA,
                    help="act+execute: max abs joint move per control step (rad)")
    ap.add_argument("--viz", dest="viz", action="store_true", default=VIZ,
                    help="act: show the 16-frame window (big current pair + 4x4 grid)")
    ap.add_argument("--no-viz-act", dest="viz", action="store_false",
                    help="act: disable the window (headless)")
    args = ap.parse_args()

    from openpi_client import websocket_client_policy

    print(f"connecting to {args.policy_host}:{args.port} (mode={args.mode}, "
          f"execute={args.execute})")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.port)
    meta = getattr(client, "_server_metadata", {})
    print("connected; server metadata:", meta)

    # The server's checkpoint config decides the capture resolution (falls back to --send_res).
    send_res = args.send_res
    if isinstance(meta, dict) and meta.get("fixed_resolution"):
        send_res = int(meta["fixed_resolution"][0])
        if send_res != args.send_res:
            print(f"using server resolution {send_res} (overrides --send_res {args.send_res})")
    cam = RobotCamera(send_res, lerobot_path=args.lerobot_path)

    if args.mode == "act":
        if meta.get("serves") != "actions":
            print("WARNING: server is not serving actions — start it with --checkpoint.")
        try:
            run_act(client, cam, execute=args.execute, meta=meta,
                    gravity_comp_time=args.gravity_comp_time,
                    max_action_delta=args.max_action_delta, viz=args.viz)
        finally:
            cam.close()
        return

    if args.mode != "sweep":
        try:
            run_live(client, cam, args.mode, args.layers, args.send_res, no_viz=args.no_viz)
        finally:
            cam.close()
        return

    history = deque(maxlen=WINDOW_FRAMES)
    rows = []
    try:
        for mode in MODES:
            for layers in LAYERS:
                history.clear()
                r = run_phase(client, cam, mode, layers, args.steps, args.warmup, history)
                print(f"  {mode:>6} L{layers:<2}: rtt={r['rtt_med']:.1f}ms server={r['server_ms']:.1f} "
                      f"(vis={r['vision_ms']:.1f} lm={r['lm_ms']:.1f}) net={r['net_ms']:.1f} "
                      f"payload={r['payload_KB']:.0f}KB {r['hz']:.1f}Hz")
                rows.append(r)
    finally:
        cam.close()

    write_results(rows, meta)


if __name__ == "__main__":
    main()
