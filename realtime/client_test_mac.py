#!/usr/bin/env python3
"""Mac webcam speed-test client for the streaming Qwen VLM server.

Self-contained: no lerobot, no openpi_client. Grabs frames from the Mac webcam (OpenCV), sends them to
the H200 server, shows the frame being sent + prints per-step latency. Robot state is sent as zeros
(speed test only). Wire-compatible with realtime/server.py (msgpack_numpy over websocket, inlined here).

Install on the Mac (once):
    pip install numpy opencv-python websockets msgpack

Run (server already up on the H200):
    python client_test_mac.py --host iris-hgx-2 --port 8000 --mode cached --layers 16
    # if the H200 isn't directly reachable, tunnel first in another terminal:
    #   ssh -N -L 8000:iris-hgx-2:8000 <you>@<login-host>
    # then:  python client_test_mac.py --host localhost --port 8000

Keys: press 'q' in the window (or Ctrl-C) to quit.
"""

from __future__ import annotations

import argparse
import functools
import statistics
import time
from collections import deque

import msgpack
import numpy as np

# --------------------------------------------------------------------------- #
# Inlined msgpack_numpy (identical wire format to the server / openpi_client). #
# --------------------------------------------------------------------------- #
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# --------------------------------------------------------------------------- #
class WSClient:
    """Minimal sync websocket client (msgpack_numpy)."""

    def __init__(self, host: str, port: int):
        from websockets.sync.client import connect

        uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._ws = connect(uri, max_size=None, compression=None, open_timeout=15)
        self._packer = Packer()
        self.metadata = unpackb(self._ws.recv())

    def infer(self, obs: dict):
        data = self._packer.pack(obs)
        self._ws.send(data)
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"server error:\n{resp}")
        return unpackb(resp), len(data)

    def close(self):
        self._ws.close()


# --------------------------------------------------------------------------- #
class WebcamSource:
    """Mac webcam via OpenCV -> RGB uint8 [send_res, send_res, 3]."""

    def __init__(self, camera_index: int, send_res: int):
        import cv2

        self.cv2 = cv2
        self.send_res = send_res
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}. Try --camera 1 etc.")
        # warm the camera (first reads are often empty)
        for _ in range(5):
            self.cap.read()

    def capture(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Webcam read failed.")
        frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        frame = self.cv2.resize(frame, (self.send_res, self.send_res))
        return frame.astype(np.uint8), np.zeros(14, dtype=np.float32)

    def close(self):
        self.cap.release()


class FakeSource:
    """Synthetic frames (no cv2/webcam) — for verifying the network path without a camera."""

    def __init__(self, send_res: int):
        self.send_res = send_res
        self.rng = np.random.default_rng(0)

    def capture(self):
        return (self.rng.integers(0, 256, (self.send_res, self.send_res, 3), dtype=np.uint8),
                np.zeros(14, dtype=np.float32))

    def close(self):
        pass


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="server host (e.g. iris-hgx-2 or localhost via tunnel)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mode", default="cached", choices=["cached", "full"])
    ap.add_argument("--layers", type=int, default=16, help="early-exit depth (1-36)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (Mac default 0)")
    ap.add_argument("--send_res", type=int, default=336)
    ap.add_argument("--steps", type=int, default=0, help="0 = run until 'q'/Ctrl-C")
    ap.add_argument("--no_viz", action="store_true", help="print latency only, no window")
    ap.add_argument("--fake_cam", action="store_true", help="synthetic frames (no webcam) to test the link")
    args = ap.parse_args()

    print(f"Connecting to ws://{args.host}:{args.port} ...")
    client = WSClient(args.host, args.port)
    print("connected; server metadata:", client.metadata)

    # Window length comes from the server (set by its --num-pairs); full mode must send exactly this many.
    WINDOW_FRAMES = int(client.metadata.get("window_frames", 30))
    n_send = 2 if args.mode == "cached" else WINDOW_FRAMES

    cam = FakeSource(args.send_res) if args.fake_cam else WebcamSource(args.camera, args.send_res)
    cv2 = None
    if not args.no_viz:
        import cv2

    history = deque(maxlen=WINDOW_FRAMES)
    win = f"qwen sent frame [{args.mode} L{args.layers}]"
    client.infer({"mode": "reset"})
    print(f"Streaming ({args.mode}, L{args.layers}). Press 'q' in the window or Ctrl-C to stop.")

    rtts = []
    step = 0
    try:
        while True:
            img, state = cam.capture()
            history.append(img)
            while len(history) < n_send:
                history.append(img)
            frames = np.stack(list(history)[-n_send:], axis=0)

            obs = {"mode": args.mode, "num_layers": args.layers, "frames": frames, "state": state}
            t0 = time.perf_counter()
            resp, payload = client.infer(obs)
            rtt = (time.perf_counter() - t0) * 1000.0
            rtts.append(rtt)

            srv = resp["server_timing"]["infer_ms"]
            vis, lm = resp.get("vision_ms", 0.0), resp.get("lm_ms", 0.0)
            ready = resp.get("ready", True)
            tag = "READY" if ready else f"warmup {resp.get('num_pairs', '?')}/15"
            print(f"step {step:4d}  rtt={rtt:6.1f}ms  server={srv:6.1f}ms (vis={vis:5.1f} lm={lm:5.1f})  "
                  f"net={rtt - srv:5.1f}ms  payload={payload/1024:.0f}KB  {1000.0/rtt:5.1f}Hz  [{tag}]")

            if not args.no_viz:
                disp = cv2.cvtColor(history[-1], cv2.COLOR_RGB2BGR)
                disp = cv2.resize(disp, (512, 512), interpolation=cv2.INTER_NEAREST)
                for i, txt in enumerate([
                    f"{args.mode}  L{args.layers}  [{tag}]",
                    f"rtt {rtt:.0f}ms   server {srv:.0f}ms",
                    f"vis {vis:.0f}  lm {lm:.0f}  net {rtt - srv:.0f}ms",
                ]):
                    y = 24 + i * 26
                    cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(win, disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            step += 1
            if args.steps and step >= args.steps:
                break
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        cam.close()
        if not args.no_viz and cv2 is not None:
            cv2.destroyAllWindows()
        client.close()

    if rtts:
        warm = rtts[15:] if len(rtts) > 20 else rtts  # drop warm-up
        print(f"\n{len(warm)} steady steps: rtt median={statistics.median(warm):.1f}ms "
              f"mean={statistics.mean(warm):.1f}ms  ~{1000.0/statistics.median(warm):.1f}Hz")


if __name__ == "__main__":
    main()
