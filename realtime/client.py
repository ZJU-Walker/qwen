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
VIZ = True                        # show the 16-frame window window (big current pair + 4x4 grid)
CONTROL_FREQ = 30.0               # send/execute rate in Hz (dt = 1/CONTROL_FREQ)
ACTIONS_PER_CHUNK = 20            # how many of the [horizon=30, 14] chunk to run before re-requesting
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
    """Mirror of the server's rolling window: a client-side deque of the last `window_frames`
    frames captured, rendered as the big current pair on top + a 4x4 grid of all 16 below.

    Purely visual (the client still only SENDS 2 frames/tick; the server caches the rest). Newest
    pair is boxed in the grid. Call push(a, b) each tick, then draw(overlay_lines); q closes it.
    """

    def __init__(self, window_frames: int, cols: int = 4, big: int = 384, cell: int = 160):
        import cv2

        self.cv2 = cv2
        self.window_frames = window_frames
        self.cols = cols
        self.rows = (window_frames + cols - 1) // cols
        self.big, self.cell = big, cell
        self.frames = deque(maxlen=window_frames)  # RGB uint8, oldest..newest
        self.win = "qwen window (sent to server)"

    def push(self, frame_a, frame_b):
        self.frames.append(np.asarray(frame_a, dtype=np.uint8))
        self.frames.append(np.asarray(frame_b, dtype=np.uint8))

    def draw(self, overlay_lines=None) -> bool:
        """Render one frame; returns False if the user pressed q (to stop)."""
        cv2 = self.cv2
        if not self.frames:
            return True
        frames = list(self.frames)
        # front-pad the display to a full window like the server does (repeat oldest)
        while len(frames) < self.window_frames:
            frames.insert(0, frames[0])

        def bgr(f, size):
            return cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (size, size),
                              interpolation=cv2.INTER_NEAREST)

        # top: the current (newest) pair, large
        top = np.hstack([bgr(frames[-2], self.big), bgr(frames[-1], self.big)])
        cv2.putText(top, "current pair (sent this tick)", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # bottom: 4x4 grid of all 16, newest two boxed
        cells = []
        for i, f in enumerate(frames):
            c = bgr(f, self.cell)
            if i >= self.window_frames - 2:  # newest pair
                cv2.rectangle(c, (1, 1), (self.cell - 2, self.cell - 2), (0, 255, 0), 3)
            cv2.putText(c, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cells.append(c)
        rows = [np.hstack(cells[r * self.cols:(r + 1) * self.cols]) for r in range(self.rows)]
        grid = np.vstack(rows)

        # align widths and stack
        W = max(top.shape[1], grid.shape[1])
        def padw(img):
            if img.shape[1] < W:
                img = np.hstack([img, np.zeros((img.shape[0], W - img.shape[1], 3), np.uint8)])
            return img
        canvas = np.vstack([padw(top), padw(grid)])

        for k, line in enumerate(overlay_lines or []):
            y = 58 + k * 26  # overlay sits below the "current pair" caption on the big top row
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(self.win, canvas)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self):
        self.cv2.destroyAllWindows()


def run_act(client, cam, execute: bool, meta: dict | None = None,
            gravity_comp_time: float = 5.0, max_action_delta: float = 0.15,
            control_freq: float = 30.0, actions_per_chunk: int = 20, viz: bool = False):
    """Stage-2 VLA loop against a --checkpoint server (mode 'act').

    Cadence is EXPLICIT (top-of-file constants), decoupled from the training grid:
    - control_freq: actions are sent at this rate (Hz); dt = 1/control_freq.
    - actions_per_chunk: how many of the server's [horizon,14] chunk to execute before requesting
      the next one. The request for chunk k+1 is fired when chunk k STARTS executing, so inference
      (~0.4 s) overlaps execution (actions_per_chunk / control_freq s). If the next chunk is late,
      the arm HOLDS the last action (chunk[min(idx,len-1)]) until it lands.

    The 2 frames per request are captured ~1/fps s apart to match the training pair grid; state is
    the current robot state at request time.

    Safety (from openpi eval_real_RTC_working_warmup.py): with --execute, a gravity-comp warmup lets
    the operator hand-place BOTH follower arms (incl. the static left arm) first; every commanded
    action is then rate-limited to +/-max_action_delta rad/step. Without --execute: prints only.
    """
    import threading

    meta = meta or {}
    fps = int(meta.get("fps", 3))
    horizon = int(meta.get("horizon", 30))
    actions_per_chunk = max(1, min(actions_per_chunk, horizon))
    frame_gap_s = 1.0 / fps                 # real-time gap between the pair's two frames
    dt = 1.0 / control_freq

    window_frames = int(meta.get("window_frames", 16))
    viewer = WindowViz(window_frames) if viz else None
    last_action = cam.gravity_comp_warmup(gravity_comp_time) if execute else None
    client.infer({"mode": "reset"})
    print(f"ACT loop: horizon={horizon}, executing {actions_per_chunk}/chunk @ {control_freq:.0f} Hz "
          f"({'EXECUTING' if execute else 'DRY RUN — prints only'}){' +viz' if viz else ''}. "
          f"Ctrl-C{' or q in window' if viz else ''} to stop.")

    result = {}  # cross-thread handoff from the request thread
    lock = threading.Lock()
    last_meta = {"rtt": 0.0, "vis": 0.0, "prefill": 0.0, "denoise": 0.0, "pairs": "?"}

    def request(frames, state):
        t0 = time.perf_counter()
        resp = client.infer({"mode": "act", "frames": frames, "state": state})
        with lock:
            result["resp"], result["rtt"] = resp, (time.perf_counter() - t0) * 1e3

    def capture_pair():
        """Two frames ~1/fps s apart + the newest state (matches the training pair)."""
        frame_a, _ = cam.capture()
        time.sleep(frame_gap_s)
        frame_b, state = cam.capture()
        if viewer is not None:
            viewer.push(frame_a, frame_b)  # mirror the sent pair into the client-side window
        return np.stack([frame_a, frame_b]), state

    def show():
        if viewer is None:
            return True
        lines = [f"tick {tick}  {control_freq:.0f}Hz  exec {actions_per_chunk}/{horizon}  "
                 f"{'EXEC' if execute else 'DRY'}",
                 f"rtt {last_meta['rtt']:.0f}ms  vis {last_meta['vis']:.0f}  "
                 f"prefill {last_meta['prefill']:.0f}  denoise {last_meta['denoise']:.0f}  "
                 f"pairs {last_meta['pairs']}"]
        return viewer.draw(lines)

    chunk, pending, tick = None, False, 0
    try:
        # Prime the first chunk synchronously so we have actions to execute.
        frames, state = capture_pair()
        request(frames, state)
        while True:
            with lock:
                resp = result.pop("resp", None)
                rtt = result.pop("rtt", None)
            if resp is not None:
                chunk, pending, tick = np.asarray(resp["actions"]), False, tick + 1
                last_meta = {"rtt": rtt, "vis": resp["vision_ms"], "prefill": resp["prefill_ms"],
                             "denoise": resp["denoise_ms"], "pairs": resp.get("num_pairs", "?")}
                print(f"tick {tick:4d}  rtt={rtt:6.1f}ms  (vis={resp['vision_ms']:.0f} "
                      f"prefill={resp['prefill_ms']:.0f} denoise={resp['denoise_ms']:.0f})  "
                      f"pairs={resp.get('num_pairs', '?')}")

            # Execute up to actions_per_chunk from the current chunk; fire the next request at idx 0.
            for idx in range(actions_per_chunk):
                t_step = time.perf_counter()
                if idx == 0 and not pending:
                    frames, state = capture_pair()
                    pending = True
                    threading.Thread(target=request, args=(frames, state), daemon=True).start()

                action = chunk[min(idx, len(chunk) - 1)]  # hold last if chunk shorter than N
                if execute:
                    if last_action is not None:
                        delta = np.clip(action - last_action, -max_action_delta, max_action_delta)
                        action = (last_action + delta).astype(np.float32)
                    last_action = action
                    cam.act(action)
                elif idx == 0:
                    print(f"    action[0:4]={np.round(action[:4], 3).tolist()} ...")
                if not show():
                    raise KeyboardInterrupt
                time.sleep(max(0.0, dt - (time.perf_counter() - t_step)))

            # If the next chunk hasn't arrived, hold the last action until it does.
            while pending:
                with lock:
                    if "resp" in result:
                        break          # arrived; the top-of-loop handler will consume it
                if execute and last_action is not None:
                    cam.act(last_action)
                if not show():
                    raise KeyboardInterrupt
                time.sleep(dt)
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
    ap.add_argument("--control-freq", type=float, default=CONTROL_FREQ,
                    help="act: send/execute rate in Hz")
    ap.add_argument("--actions-per-chunk", type=int, default=ACTIONS_PER_CHUNK,
                    help="act: how many of the chunk to execute before requesting the next")
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
                    max_action_delta=args.max_action_delta,
                    control_freq=args.control_freq,
                    actions_per_chunk=args.actions_per_chunk,
                    viz=args.viz)
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
