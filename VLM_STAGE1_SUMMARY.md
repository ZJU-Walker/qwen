# Stage 1 Summary — Streaming Qwen2.5-VL-3B Context-Feature Module

Context handoff document: what is already built and verified, so Stage 2 (the action expert /
full VLA) can be planned against it. Code lives at `/iris/projects/humanoid/qwen/`
(package `src/streaming_qwen_vlm/`), runs on the Iris H200 (`iris-hgx-2`), env
`/iris/u/kewalk/.conda/envs/qwen3vl/bin/python` (torch 2.6.0, transformers 4.57.6, flash_attn 2.7.4).

## 1. What Stage 1 is

A **streaming VLM feature extractor** for a robotics VLA, wrapping the **stock pretrained
Qwen2.5-VL-3B-Instruct** (no fine-tuning, no custom checkpoint; loaded via `from_pretrained`).
Per control step it consumes:

- the newest **2 camera frames** (rolling 30-frame / 10 s history at 3 fps, cam_high 960x540
  squashed to a fixed 336x336),
- a **fixed language instruction** (baked into a cached chat-template prompt),
- a **14-dim robot state vector**,

and returns **context-token hidden states** for an action expert to cross-attend to. It does NOT
generate text, act, or train — Stage 1 is inference-time feature extraction only.

## 2. Output interface (what the action expert receives)

`vlm.step(pair, state) -> VLMOutput | None` (None during the 14-step warm-up while the window fills):

- `context_tokens`: `[1, S, 2048]`, bf16, layout **[video | instruction | state]**
  - video: **2160 tokens** (15 temporal pairs x 144 tokens; each pair = 2 frames -> one 12x12
    merged grid at 336x336)
  - instruction: ~**28 tokens** (chat scaffold + instruction text; fixed per instruction)
  - state: **1 token** (configurable `num_state_tokens`)
  - total S ≈ 2189
- `token_types`: `[1, S]` with 0=text, 1=video, 2=state; `context_mask`: `[1, S]` all-ones
- `layer_index`: which LLM layer the features come from (the early-exit depth N)

Features are the hidden states after running only the **first N of 36 decoder layers**
(N configurable 1–36; final RMSNorm applied by default, switchable).

## 3. How it stays fast — three verified mechanisms

1. **Two-frame vision caching** (`vision_cache.py`). Each new 2-frame pair is encoded ONCE by the
   vision tower (`get_video_features`) into a `[144, 2048]` embedding; a rolling deque of 15 pair
   embeddings is concatenated to reconstruct the full `[2160, 2048]` window feature. The vision
   encoder runs on 2 frames per step, never 30. Valid because Qwen's vision attention is blocked
   per tower call (`cu_seqlens`), verified empirically in fp32 (max abs diff 1.4e-4 vs full-window
   encode).
2. **Embedding injection** (`model.py`). Build `inputs_embeds` from the fixed prompt template,
   `masked_scatter` the cached video features into the `video_token_id` slots, run the LM with no
   `pixel_values_videos` (vision tower skipped). Exactly matches the reference forward
   (max abs diff 0.0).
3. **Real early exit** (`early_exit.py`). The decoder loop is genuinely truncated to N layers
   (real compute savings), reusing the official mask builders / mrope. Exactly matches
   `hidden_states[N]` of the official forward (max abs diff 0.0).

Other structural choices: the whole window is ONE logical video (grid_thw `[15,12,12]`) with mrope
temporal positions (`second_per_grid_ts = 2/3 s`); resolution pinned via `min_pixels==max_pixels`
so every pair has an identical token layout; prompt template, positions, and masks are all
precomputed once. **No LM KV-cache reuse across steps** — every step is a full prefill over
~2189 tokens (deliberately out of scope for Stage 1).

## 4. Frozen vs trainable

- **Frozen / pretrained**: everything from Qwen — vision tower, embeddings, all 36 decoder layers.
- **Not pretrained (placeholder)**: `StateProjector` (`state_encoder.py`) — a small MLP
  `14 -> 256 -> 2048` (Linear+GELU+Linear), deterministically seeded random init. It maps robot
  state into an input token appended after the prompt. It is expected to be trained in Stage 2.

## 5. Measured performance envelope (H200)

Target was 1.5 Hz (667 ms); everything is far under it. Steady-state per-step latency
(one pair encode + full-window LM forward), peak GPU ~7.5 GB:

| early-exit N | vision ms | LM ms | total ms | Hz |
|---|---|---|---|---|
| 8  | 34 | 12 | 46 | 21.6 |
| 16 | 34 | 20 | 55 | 18.3 |
| 36 | 23 | 41 | 64 | 15.7 |

**Realtime client/server path** (`realtime/`): websocket + msgpack_numpy server on the H200,
wire-compatible with OpenPI's `WebsocketClientPolicy`. Loopback-verified end-to-end: cached mode at
N=16 runs ~15 Hz with a 662 KB/step payload (vs 9.9 MB and ~6 Hz if the whole 30-frame window is
re-sent and re-encoded). Real-robot (workstation -> H200 over network) sweep not yet run.
Returning the full context tensor over the wire costs ~9 MB/step (fp16) — the action expert should
live on the same GPU as the VLM, or features must be compressed.

## 6. Verification status

6 pytest tests, all passing on the H200, run in gating order
(`PYTHONPATH=src python -m pytest tests/ -v -s`): single-logical-video layout, fp32 pair-cache
equivalence (GATING), injection equivalence (exact), early-exit equivalence (exact), sliding-window
order, output shapes. A real-data demo (`examples/run_episode.py`) streams Trossen cam_high
episodes (LeRobot format, `/iris/projects/humanoid/trossen_data/0528_merge_block_mem`, 14-dim
`observation.state`, episodes ~6–7 s).

## 7. Explicitly deferred to Stage 2 (the plan to be made)

- **Action expert**: architecture (e.g. flow-matching / diffusion DiT a la pi0, cross-attending to
  `context_tokens`), size, action horizon/chunking, action space for the Trossen bimanual arms
  (14-dim state suggests 2x 6-DoF + 2 grippers).
- **Training**: which parts unfreeze (StateProjector at minimum; LoRA/full FT of Qwen?), loss,
  data pipeline over the LeRobot dataset, whether early-exit depth N is fixed during training
  (features differ per N — the expert must train against the depth it will run at).
- **Open design questions**: choice of N (speed/quality trade-off unmeasured on task success);
  1 state token enough?; whether to add KV-cache reuse across steps later; whether features
  should be pooled/compressed before cross-attention; sync vs async VLM/expert rates
  (VLM ~15–20 Hz available, control may want faster expert ticks on stale context).
