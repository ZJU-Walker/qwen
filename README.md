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
out = None
# Feed two new frames per step; returns None during warm-up until the 30-frame window fills.
for pair in stream_of_frame_pairs:           # each pair: [frame_a, frame_b] HxWx3 uint8 RGB
    out = vlm.step(pair, state)
    if out is not None:
        ctx = out.context_tokens              # [1, S, 2048]; S = 2160 + instr_len + num_state_tokens
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
