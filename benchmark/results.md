# H200 Benchmark — Streaming Qwen2.5-VL-3B feature module

Steady-state per-`step()` latency (one new 2-frame pair encoded + 30-frame window LM forward
with real early exit) and peak memory. Target: `total_update_ms < 667` (1.5 Hz).

| num_llm_layers_to_run | fixed_resolution | video_tokens | vision_encode_ms | lm_forward_ms | total_update_ms | peak_alloc_MB | peak_reserved_MB | sustained_Hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 336x336 | 2160 | 34.09 | 12.12 | 46.21 | 7511.5 | 7804.0 | 21.64 |
| 12 | 336x336 | 2160 | 34.8 | 15.7 | 50.5 | 7511.5 | 7804.0 | 19.8 |
| 16 | 336x336 | 2160 | 34.4 | 20.16 | 54.56 | 7511.5 | 7804.0 | 18.33 |
| 24 | 336x336 | 2160 | 27.6 | 27.24 | 54.84 | 7511.5 | 7804.0 | 18.23 |
| 36 | 336x336 | 2160 | 22.96 | 40.65 | 63.61 | 7511.5 | 7804.0 | 15.72 |

## Correctness (first H200 run)
- Vision pair-cache equivalence (fp32): max abs diff 1.41e-4, mean abs diff 8.06e-7 (bf16: max 7.79, mean 2.62e-2). Validates the two-frame caching design.
- Cached-injection vs full forward: max abs diff 0.0, cosine 1.0 (exact).
- Early exit vs official text-model forward (N=8,16 pre-norm; N=36+final_norm): max abs diff 0.0 (exact).
