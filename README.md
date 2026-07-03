# Streaming Qwen2.5-VL-3B VLM Context-Feature Module

Stage 1 of a robotics VLA. This package wraps **Qwen2.5-VL-3B-Instruct** as a streaming feature
module: it consumes a rolling 30-frame visual history + a fixed language instruction + a robot-state
vector and returns **context-token hidden states** for a future action expert to cross-attend to.

It does NOT generate text, build an action expert, train, or control a robot. See `claude_plan.md`
for the full spec and `qwen_vlm_streaming_plan.md` for the original design notes.

## Key ideas

- **Two-frame vision caching** (`vision_cache.py`): each new 2-frame pair is encoded ONCE via
  `get_video_features` into a `[144, 2048]` embedding. A rolling deque of 15 pair-embeddings is
  concatenated to reconstruct the full `[2160, 2048]` window feature — the vision encoder runs on 2
  frames per step, never on the whole window. Validated to fp32 precision against a full-window encode
  (test #2).
- **Injection seam** (`model.py`): build `inputs_embeds`, `masked_scatter` the cached video features
  into the `video_token_id` slots, then run the LLM with NO `pixel_values_videos` so the vision tower
  is not re-run. Matches the reference forward exactly (test #3, max abs diff 0.0).
- **Real early exit** (`early_exit.py`): the decoder loop is truncated to `N` layers (genuine compute
  savings), reusing the official mask builders / rotary embedding. NOT post-hoc `hidden_states[N]`
  indexing. Matches the official forward exactly (test #4, max abs diff 0.0).

## Environment

Use the prebuilt conda env directly (no `conda activate`):

```
/iris/u/kewalk/.conda/envs/qwen3vl/bin/python
```

torch 2.6.0, transformers 4.57.6, flash_attn 2.7.4. `pytest` is required for the tests
(`pip install pytest` into that env if missing).

## Usage

```python
import numpy as np
from streaming_qwen_vlm import StreamingQwenVLM, VLMConfig

cfg = VLMConfig(num_llm_layers_to_run=16)   # early-exit depth (1..36)
vlm = StreamingQwenVLM(cfg)

state = np.zeros(cfg.state_dim, dtype=np.float32)
# Feed two new frames per step. Output from tick 0: until the 30-frame window fills, it is
# front-padded by repeating the oldest pair (Stage-2 D6; no more warm-up None phase).
for pair in stream_of_frame_pairs:           # each pair: [frame_a, frame_b] HxWx3 uint8 RGB
    out = vlm.step(pair, state)
    ctx = out.context_tokens                  # [1, S, 2048]; S = 2160 + instr_len + num_state_tokens
    # out.token_types: 0=text, 1=video, 2=state ; out.context_mask: [1, S]
```

Context-token layout is `[video | instruction | state]` (documented in `model.py`).

## Tests

Run in the gating order (trust streaming `step()` only after the gating vision-cache test passes):

```
cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
  /iris/u/kewalk/.conda/envs/qwen3vl/bin/python -m pytest tests/ -v -s
```

| # | test | checks |
|---|------|--------|
| 1 | `test_single_logical_video` | window is one `[15,24,24]` video → 2160 video tokens |
| 2 | `test_vision_cache_equivalence` | **GATING (fp32)** pair-concat == full-window encode |
| 3 | `test_cached_injection_equivalence` | scatter-injection == full forward with pixels |
| 4 | `test_early_exit_equivalence` | truncated loop == official forward `hidden_states[N]` |
| 5 | `test_sliding_window` | pair cache shifts to the latest 15 pairs in order |
| 6 | `test_output_shape` | output shapes + token-type counts across N |

## Run on real data (Trossen cam_high)

Stream a real `cam_high` episode (frames + 14-dim state) from a LeRobot-format dataset:

```
cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
  /iris/u/kewalk/.conda/envs/qwen3vl/bin/python examples/run_episode.py \
    --root /iris/projects/humanoid/trossen_data/0528_merge_block_mem --episode 0 --layers 16
```

Reads pre-extracted JPGs from `frames/cam_high/episode_*/` and `observation.state` from the parquet.
cam_high is 960x540 (16:9); the default config squashes it to the square 336x336 grid. Episodes here
are ~6-7 s, so the demo evenly subsamples `window_frames` frames to fill the 10 s window from a single
episode (pass `--stride` for fixed native-rate sampling).

## Benchmark

```
cd /iris/projects/humanoid/qwen && PYTHONPATH=src \
  /iris/u/kewalk/.conda/envs/qwen3vl/bin/python benchmark/benchmark_h200.py
```

Sweeps `N ∈ {8,12,16,24,36}` and writes `benchmark/results.md`. On the H200, all depths are far under
the 667 ms / 1.5 Hz target (full 36 layers ≈ 64 ms ≈ 15.7 Hz; N=8 ≈ 46 ms ≈ 21.6 Hz), peak ≈ 7.5 GB.

---

# Stage 2 — VLA: joint-attention action expert + FAST AR training

Implements the locked design in `claude_plan.md` (D1–D8), π0.5-faithful:

- **Expert** (`expert.py`, `flow.py`): 36-layer / width-1024 parallel transformer; at every layer
  the 30 noisy-action suffix tokens attend over `[prefix K_ℓ,V_ℓ (Qwen) ⊕ suffix K/V]` in one
  softmax (heads 16/2/128 matched to Qwen). adaRMS(τ) with zero-init modulation; flow matching
  (τ~Beta(1.5,1), 10 Euler steps). ≈850M params.
- **KV export** (`llm_forward.py`): train-capable manual decoder loop; per-layer post-rope prefix
  K/V recomputed from the layer inputs under `torch.no_grad()` — that IS the knowledge-insulation
  boundary (backbone trains only via the AR loss). `early_exit.py` untouched (ablations).
- **Prompt** (`prompt_builder.build_act_template`): `[scaffold | 2160 video | instruction |
  "\nState:" | 56 state ids | assistant tail]` — state rendered as constant-length 3-digit bins
  (`normalize.py`, `state_text.py`), so `S_prefix` is a project-wide constant. StateProjector retired
  (`num_state_tokens=0`).
- **FAST side-car** (`fast_tokens.py`, `vla.py`): Qwen vocab NOT resized (tied embeddings);
  FAST ids live at `V_BASE+` with separate `fast_embed`/`fast_head`; `L = L_AR + L_flow`.
- **Data** (`training/dataset.py`): any control step t≥10; pair grid = frames (20k, 20k+10);
  windows front-padded identically to streaming; chunks `a[t:t+30]` hold-pose padded. Unique
  pairs are ViT-encoded once per sample and gathered into window slots via `slot_map` (episodes
  are shorter than the 15-pair window, so this cuts vision compute ~3x; gradients identical by
  linearity).
- **Deviations from claude_plan.md** (deliberate): vocab side-car instead of resize; hold-pose
  padding instead of loss masks; EMA on expert only; t≥10 sampling floor; adaRMS final norm.

## Run order (all commands assume `conda activate qwen3vl` first)

Every block below also assumes:

```bash
cd /iris/projects/humanoid/qwen
export PYTHONPATH=src:.
```

### Step 0 — one-time setup (H200, needs internet)

```bash
pip install scipy wandb protobuf
# download the FAST tokenizer (prints its vocab size; ~small download, cached under ~/.cache/huggingface)
python -c "from transformers import AutoProcessor; p=AutoProcessor.from_pretrained('physical-intelligence/fast', trust_remote_code=True); print(p.vocab_size)"
wandb login
```

### Step 1 — M1 gates (data/interface layer)

```bash
# 1. compute q01/q99 normalization stats on the train split (writes checkpoints/norm_stats.json)
python -m streaming_qwen_vlm.normalize --root /iris/projects/humanoid/trossen_data/0528_merge_block_mem --out checkpoints/norm_stats.json

# 2. measure FAST tokens/chunk (already done: stride-1 max=147, p99=134 -> budget 192 in config)
python -m streaming_qwen_vlm.fast_tokens --measure --stats checkpoints/norm_stats.json

# 3. tests — CPU-ok (front-padding semantics; FAST round-trip needs the step-0 download)
python -m pytest tests/test_front_padding.py tests/test_fast_roundtrip.py -v -s

# 4. tests — need the H200 (loads the 3B model): KV export + dataloader<->streaming parity
python -m pytest tests/test_kv_export_equivalence.py tests/test_dataloader_streaming_parity.py -v -s

# 5. updated Stage-1 tests (emit-from-tick-0)
python -m pytest tests/test_output_shape.py tests/test_sliding_window.py -v -s
```

### Step 2 — M2 gates (model)

```bash
# 1. expert unit tests (CPU-ok)
python -m pytest tests/test_expert_attention.py tests/test_expert_init.py -v -s

# 2. overfit-one-episode gate (H200, ~2k steps)
python -m streaming_qwen_vlm.training.train --overfit-episode 0 --steps 2000 --out-dir checkpoints/overfit

# 3. visual check: replay the overfit policy on the same episode
python examples/replay_policy.py --checkpoint checkpoints/overfit/step_002000 --episode 0
```

Gate passes when FAST token accuracy → ~1.0, val open-loop MSE ≈ 0, and the replay curves track
the demo.

### Step 3 — M3 full training (H200, ~30k steps)

```bash
# run3 config: 224x224, 16-frame window @ 2 fps (512 video tokens, ~3x fewer FLOPs than 336/30f/3fps)
python -m streaming_qwen_vlm.training.train --out-dir checkpoints/run3 --steps 15000 \
    --resolution 224 --fps 2 --num-pairs 8 --ckpt-stride 2 --compile
# resume after interruption:  add --resume
# overfitting fallback:       add --expert-width 768
# on a shared node: prefix CUDA_VISIBLE_DEVICES=<gpu>, consider --num-workers 8
```

The vision config (`--resolution/--fps/--num-pairs`) is saved in each checkpoint's meta.json;
`ActPolicy`, `replay_policy.py`, and the realtime server/client all rebuild the grid from the
checkpoint (the client reads the server's metadata for capture resolution and tick cadence), so
checkpoints from DIFFERENT vision configs (e.g. run1 at 336/3fps and run3 at 224/2fps) are all
servable with the same code.

Speed knobs: `--ckpt-stride 2` recomputes only alternate LLM layers (~+10%, +~29 GB → ~125 GB
peak); `--compile` regionally torch.compiles the 36 decoder layers (~+15-30%, first step takes
minutes to warm up; if it misbehaves just relaunch without it). Smoke a flag combo first:

```bash
python -m streaming_qwen_vlm.training.train --overfit-episode 0 --steps 60 --warmup-steps 10 \
    --eval-every 30 --save-every 30 --keep-every 60 --out-dir checkpoints/smoke2 --ckpt-stride 2 --compile
```

Checkpoint cadence (defaults): deployable `step_XXXXXX` snapshots (11 GB) every 500 steps;
snapshots at multiples of `--keep-every` (2000) are kept forever, others pruned to the newest
`--keep-last` (3); the fp32 resume bundle (`latest/`, 55 GB) refreshes every `--resume-every`
(2000). Disk math at 30k steps ≈ 165 GB milestones + 33 GB recents + 55 GB resume — watch
`df -h /iris/projects/humanoid` and delete stale run/smoke dirs.

### Step 4 — M4 deployment

```bash
# H200: action server (with --checkpoint it serves actions, not context features)
python realtime/server.py --host 0.0.0.0 --port 8000 --checkpoint checkpoints/run1/step_030000

# robot workstation (its own env, needs openpi_client): dry-run first, add --execute to actuate
python realtime/client.py --policy_host <H200-host> --port 8000 --mode act
python realtime/client.py --policy_host <H200-host> --port 8000 --mode act --execute
```

Per tick: pair encode → state-in-prompt → 36-layer prefill (KV export, once) → 10 expert Euler
steps on the cached prefix K/V → `[30, 14]` absolute joint targets (~1.7 KB). Expected ≈105 ms
per 667 ms tick.
