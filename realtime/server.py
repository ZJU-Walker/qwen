"""Streaming Qwen VLM inference server (runs on the H200).

Wraps StreamingQwenVLM in an openpi-style policy and serves it over websocket. Each request is one
inference step; the server holds the rolling pair-cache across requests (stateful streaming), so the
"cached" mode measures the real two-frame-cached per-step cost. The "full" mode re-encodes the whole
30-frame window every request (baseline). Per-request early-exit depth via obs["num_layers"].

Run:
    cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
      /iris/u/kewalk/.conda/envs/qwen3vl/bin/python realtime/server.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streaming_qwen_vlm import StreamingQwenVLM, VLMConfig  # noqa: E402
from streaming_qwen_vlm.early_exit import run_language_early_exit  # noqa: E402
from streaming_qwen_vlm.preprocess import (  # noqa: E402
    frames_to_video_inputs,
    pair_to_pixel_values,
    prepare_frames,
)
from streaming_qwen_vlm.timing import cuda_timer  # noqa: E402
from streaming_qwen_vlm.vision_cache import PairVisionCache  # noqa: E402

from realtime.ws_server import WebsocketPolicyServer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("qwen_server")


class StreamingQwenVLMPolicy:
    """openpi-style policy: infer(obs:dict)->dict. obs['mode'] in {reset, cached, full}."""

    def __init__(self, cfg: VLMConfig, return_features: bool = False) -> None:
        self.cfg = cfg
        self.vlm = StreamingQwenVLM(cfg)
        self.return_features = return_features
        logger.info("Model loaded; ready to serve.")

    # ---- shared LLM half: inject video embeds (+state) and run N layers, return hidden states ----
    @torch.inference_mode()
    def _run_llm(self, video_embeds: torch.Tensor, state, num_layers: int) -> torch.Tensor:
        vlm = self.vlm
        inputs_embeds = vlm.model.get_input_embeddings()(vlm._input_ids)
        inputs_embeds = inputs_embeds.masked_scatter(
            vlm._video_mask.unsqueeze(-1), video_embeds.to(vlm.device, vlm.dtype).reshape(-1)
        )
        if vlm.state_proj is not None:  # Stage-1 ablation path (num_state_tokens > 0)
            state_t = torch.as_tensor(np.asarray(state, dtype=np.float32))
            state_tok = vlm.state_proj(state_t).to(vlm.device, vlm.dtype)
            inputs_embeds = torch.cat([inputs_embeds, state_tok], dim=1)
        return run_language_early_exit(
            vlm.model, inputs_embeds, vlm._attn_full, vlm._position_ids, num_layers, vlm.cfg.early_exit_norm
        )

    @torch.inference_mode()
    def _cached_step(self, frames, state, num_layers):
        vlm, cfg = self.vlm, self.cfg
        prepared = prepare_frames(list(frames), cfg)
        pinp = pair_to_pixel_values(vlm.processor, prepared, cfg)
        with cuda_timer() as t_vis:
            z = vlm.cache.encode_pair(vlm.model, pinp["pixel_values_videos"], pinp["video_grid_thw"])
            vlm.cache.push(z.to(vlm.dtype))
        # D6 front-padding: the window is padded with the oldest pair until full — ready from tick 0.
        with cuda_timer() as t_lm:
            hs = self._run_llm(vlm.cache.concat(pad_to_full=True), state, num_layers)
        resp = self._finish(hs, t_vis(), t_lm())
        resp["num_pairs"] = len(vlm.cache)
        return resp

    @torch.inference_mode()
    def _full_step(self, frames, state, num_layers):
        vlm, cfg = self.vlm, self.cfg
        prepared = prepare_frames(list(frames), cfg)
        vinp = frames_to_video_inputs(vlm.processor, prepared, cfg)
        pv = vinp["pixel_values_videos"].to(vlm.device)
        grid = vinp["video_grid_thw"].to(vlm.device)
        with cuda_timer() as t_vis:
            video_embeds = vlm.model.get_video_features(pv, grid)[0]
        with cuda_timer() as t_lm:
            hs = self._run_llm(video_embeds, state, num_layers)
        return self._finish(hs, t_vis(), t_lm())

    def _finish(self, hs: torch.Tensor, vision_ms: float, lm_ms: float) -> dict:
        # Build context tokens in [video | instruction | state] order (same as VLMOutput).
        vlm = self.vlm
        hs_t = hs[:, : vlm._S_template, :]
        video_hs = hs_t[vlm._video_mask]
        instr_hs = hs_t[vlm._text_mask]
        n_state = vlm.cfg.num_state_tokens
        parts = [video_hs, instr_hs]
        if n_state > 0:
            parts.append(hs[:, vlm._S_template :, :].reshape(n_state, -1))
        ctx = torch.cat(parts, dim=0).unsqueeze(0)  # [1, S_context, 2048]
        resp = {
            "ready": True,
            "vision_ms": float(vision_ms),
            "lm_ms": float(lm_ms),
            "feature_shape": list(ctx.shape),
            "feature_norm": float(ctx.float().norm().item()),
        }
        if self.return_features:
            resp["features"] = ctx.float().to(torch.float16).cpu().numpy()  # fp16 to halve payload
        return resp

    def infer(self, obs: dict) -> dict:
        mode = obs.get("mode", "cached")
        if mode == "reset":
            self.vlm.cache = PairVisionCache(self.cfg.num_pairs, self.cfg.tokens_per_pair)
            return {"ok": True, "mode": "reset"}

        num_layers = int(obs.get("num_layers", self.cfg.num_llm_layers_to_run))
        frames = np.asarray(obs["frames"])
        state = obs.get("state", np.zeros(self.cfg.state_dim, dtype=np.float32))

        if mode == "cached":
            resp = self._cached_step(frames, state, num_layers)
        elif mode == "full":
            resp = self._full_step(frames, state, num_layers)
        else:
            raise ValueError(f"unknown mode {mode!r} (expected reset/cached/full)")

        resp["mode"] = mode
        resp["num_layers"] = num_layers
        return resp


class ActPolicyServerAdapter:
    """Stage-2 action server: obs['mode'] in {reset, act}. Returns [horizon, 14] action chunks.

    Mutually exclusive with the Stage-1 feature modes: pass --checkpoint to serve actions from a
    fine-tuned VLA checkpoint; without it the server serves context features (cached/full).
    """

    def __init__(self, checkpoint_dir: str, device: str = "cuda") -> None:
        from streaming_qwen_vlm.policy import ActPolicy  # noqa: E402 (heavy import, act mode only)

        self.policy = ActPolicy(checkpoint_dir, device=device)
        logger.info("VLA checkpoint loaded from %s; serving actions.", checkpoint_dir)

    def infer(self, obs: dict) -> dict:
        mode = obs.get("mode", "act")
        if mode == "reset":
            self.policy.reset()
            return {"ok": True, "mode": "reset"}
        if mode != "act":
            raise ValueError(f"unknown mode {mode!r} (this server runs with --checkpoint: reset/act)")
        frames = np.asarray(obs["frames"])
        state = np.asarray(obs.get("state", np.zeros(self.policy.cfg.state_dim, dtype=np.float32)))
        out = self.policy.act(frames, state)
        t = out["timings"]
        return {
            "mode": "act",
            "ready": True,
            "actions": out["actions"],                    # float32 [horizon, 14] raw joint targets
            "num_pairs": out["num_pairs"],
            "vision_ms": t["vision_ms"],
            "prefill_ms": t["prefill_ms"],
            "denoise_ms": t["denoise_ms"],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to a training step_XXXXXX checkpoint dir -> serve ACTIONS (mode 'act') "
                         "instead of context features.")
    ap.add_argument("--num-pairs", type=int, default=15,
                    help="history window length in pairs (window_frames = 2*num_pairs). Default 15 = 30 frames; "
                         "use 5 for a 10-frame window. Client and server must agree (full mode sends 2*num_pairs frames).")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--return-features", action="store_true",
                    help="Include the fp16 context-token tensor in the response (measures its net cost).")
    args = ap.parse_args()

    if args.checkpoint:
        policy = ActPolicyServerAdapter(args.checkpoint, device=args.device)
        cfg = policy.policy.cfg
        metadata = {
            "model": cfg.model_id,
            "serves": "actions",
            "horizon": policy.policy.expert_cfg.horizon,
            "action_dim": policy.policy.expert_cfg.action_dim,
            "window_frames": cfg.window_frames,
            "fixed_resolution": list(cfg.fixed_resolution),
            "state_dim": cfg.state_dim,
        }
        server = WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata)
        logger.info("Serving ACTIONS on %s:%d (checkpoint=%s)", args.host, args.port, args.checkpoint)
        server.serve_forever()
        return

    cfg_kwargs = dict(device=args.device, num_pairs=args.num_pairs)
    if args.instruction:
        cfg_kwargs["instruction"] = args.instruction
    cfg = VLMConfig(**cfg_kwargs)

    policy = StreamingQwenVLMPolicy(cfg, return_features=args.return_features)
    metadata = {
        "model": cfg.model_id,
        "serves": "features",
        "window_frames": cfg.window_frames,
        "fixed_resolution": list(cfg.fixed_resolution),
        "total_video_tokens": cfg.total_video_tokens,
        "state_dim": cfg.state_dim,
    }
    server = WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata)
    logger.info("Serving on %s:%d (return_features=%s)", args.host, args.port, args.return_features)
    server.serve_forever()


if __name__ == "__main__":
    main()
