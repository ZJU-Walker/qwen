# Stage 2 — Locked Design (supersedes the exploratory plan)

Decisions made interactively, 2026-07-01. This is the implementation reference. π0.5 conventions below are copied from openpi source (`pi0.py`, `pi0_config.py`, `gemma.py`), not from paper memory.

## 1. Decision record

**D1 — Expert wiring: joint per-layer attention (π0.5-faithful).** The action expert is a parallel transformer with depth equal to the backbone path it attends to. At every layer ℓ, expert suffix queries attend over `[prefix K_ℓ, V_ℓ (Qwen) ⊕ suffix K_ℓ, V_ℓ (expert)]`, one softmax over the concatenated sequence. Constraint (openpi asserts it): expert must match Qwen's `head_dim`, `num_heads`, `num_kv_heads`; only width and MLP shrink. Rejected alternative: cross-attention DiT on layer-N `context_tokens` (GR00T-style) — kept on record as fallback only.

**D2 — Backbone regime: knowledge insulation, full fine-tune (π0.5-faithful).** Qwen (ViT + LLM, all 36 layers) trains with autoregressive losses only — FAST-tokenized actions. The flow expert trains with flow matching on per-layer K/V that are `detach()`ed at the backbone boundary. Rationale: flow gradients from a from-scratch expert degrade pretrained representations and slow convergence (the KI finding); next-token prediction is the loss type the VLM was built on. Rejected: frozen backbone (weaker features), flow-gradient full FT (the thing π0.5 exists to avoid, worst-case here given from-scratch expert + small data).

**D3 — State: discretized into the prompt (π0.5-faithful; correct *because* of D2).** Per-dim quantile normalization, 256 bins, bin indices rendered as text after the instruction. Prefix layout: `[video 2160 tok | instruction | state ≈30–45 tok]`. The backbone learns to condition FAST predictions on state via the AR loss. `StateProjector` is retired. Caching impact: video pair cache and instruction template unchanged; only state tokens re-embed per tick (trivial). Note: with a frozen backbone this option would have been wrong (no training signal shapes how the LM reads state); it is right only in combination with D2.

**D4 — Expert depth: N = 36, full depth.** Early exit is retired from the main path (keep the code for ablations). Both original motivations died with D2: latency headroom is enormous (64 ms full-depth vs 667 ms tick), and the AR loss shapes all 36 layers for action prediction, which is exactly the backbone π0.5 attaches its expert to.

**D5 — Actions: 30 Hz, H = 30 (1 s chunks).** Dataset control rate 30 Hz. Actions = the dataset `action` field as-is (absolute joint-position targets, 2×(6+gripper) = 14-dim, no delta conversion — confirm leader-target convention in the data). Normalization: per-dim q01/q99 → [−1, 1], computed once on the train split, shared by the flow target and the FAST tokenizer input. Execution: open-loop; a tick lands every 20 control steps, so execute 20 of 30 actions and swap (10-action buffer for late ticks). No temporal ensembling.

**D6 — Window: keep 30 frames / 10 s, front-pad, emit from tick 0.** At tick k < 14 the window is `[pair₀ × (14−k), pairs 0..k]`; mask all-ones, mrope grid unchanged. Fixes the blocker (episodes are 6–7 s < 10 s window; the old warm-up would emit nothing on this dataset and be blind for 10 s in deployment). The training dataloader applies bit-identical padding — enforced by a gating test (§5). Planned ablation: window = 1 pair, to measure rather than assume the temporal-context benefit.

**D7 — Language: FAST-only AR for v1.** Dataset has one task string per episode, no per-step subtask labels. π0.5's hierarchical subtask prediction is deferred; when subtask annotations exist, subtask text inserts between state and FAST tokens with no architecture change.

**D8 — Vision tower: trained.** Full-faithful. Consequence: nothing is precomputable during training (all 15 pairs ViT-encoded in-graph per sample). Inference pair-caching is unaffected — its validity is structural (blocked attention via `cu_seqlens`), not weight-dependent.

## 2. Model specification

Backbone: Qwen2.5-VL-3B-Instruct, all 36 layers, vocab extended by the FAST token set (~2k new rows in embedding table and LM head; Qwen lacks a large reserved tail, so resize rather than remap).

Expert (verify Qwen numbers against the loaded `config.json`):

| Field | Value |
|---|---|
| depth / width | 36 / 1024 (fallback 768 ≈ 500M if overfitting) |
| heads Q / KV / head_dim | 16 / 2 / 128 — must equal Qwen |
| MLP | SwiGLU, hidden 4096 |
| norms | RMSNorm; adaRMS conditioned on τ: zero-init Dense(width→3·width) per norm giving (scale, shift, gate); `norm(x)·(1+scale)+shift`, residual × gate |
| time path | `posemb_sincos(τ)` (min_period 4e-3, max_period 4.0) → Linear → swish → Linear → swish |
| suffix | 30 noisy-action tokens only (no state token) |
| in / out | `action_in_proj: Linear(14→1024)`; final RMSNorm → `action_out_proj: Linear(1024→14)` |
| positions | continue after prefix; Qwen rotary, mrope text mode (same index all 3 axes) |
| init | lecun-normal; modulation Denses zero (expert starts near-identity) |
| params | ≈ 850M (≈620M core + ≈230M adaRMS modulation) |

Attention structure at training: prefix causal (stock Qwen); FAST tokens are the causal continuation of the prefix; expert suffix attends to the full prefix plus bidirectionally within itself, and **never to the FAST tokens** (they do not exist at inference). At inference: no FAST tokens are decoded; prefix K/V computed once per tick and reused across denoise steps.

Flow matching (exact openpi conventions):

```
τ  ~ Beta(1.5, 1) · 0.999 + 0.001         # τ=1 is pure noise
x_τ = τ·ε + (1−τ)·a,   ε ~ N(0, I)
u   = ε − a
L_flow = mean ‖v_θ(x_τ, τ, sg[prefix K/V]) − u‖²   # sg = stop-gradient / detach
Inference: 10 Euler steps, x ← x + dt·v, dt = −1/10, τ: 1 → 0
```

## 3. Training

Total loss `L = L_AR + L_flow`, equal weights to start; single phase; loss applied only to FAST tokens (AR side) and velocity outputs (flow side), never to prompt tokens.

Sampling: sample any control step t (not just ticks — 6.5 s episodes give ~195 steps vs ~10 ticks, a ~20× multiplier). Window = pairs ending at or before t, front-padded per D6; visual staleness up to 2/3 s matches deployment's within-chunk staleness by construction. Targets: normalized `a[t : t+30]`, loss-masked past episode end; the same chunk feeds the FAST tokenizer for the AR targets. Prompt template cache keyed by the episode's `task` string.

Optimizer: AdamW, grad-clip 1.0, cosine with ~1k warmup, ~30k steps. Parameter groups: backbone (ViT + LLM) peak 2e-5; new FAST embeddings + expert peak 1e-4. Batch 8–16 with accumulation to effective 32–64, bf16 activations + fp32 master, gradient checkpointing, EMA 0.999. Memory: ≈73 GB states before activations on the 141 GB H200 — fits. Ballpark 1–3 days for the run.

Mandatory gate before the full run: overfit one episode — 10-step denoised open-loop MSE ≈ 0, replayed trajectory visually tracks the demo, FAST token accuracy → ~1. Exercises padding, masks, positions, adaRMS, both losses.

Skipped for v1, revisit conditions on record: web/VQA co-training (add if open-instruction generalization starts to matter); subtask AR (add when labels exist).

## 4. Deployment

Expert co-located with Qwen in the realtime server (structural under D1; also removes the 9 MB/step feature-transfer cost — the chunk is ~1.7 KB). Per tick: recv `(2 frames, raw state)` → pair encode into rolling cache → discretize state with stored quantile stats into prompt → 36-layer prefill (once) → 10 expert denoise steps on cached prefix K/V → return `[30, 14]` chunk. Client executes 20 actions at 30 Hz until the next chunk. Latency ≈ 34 + 41 + ~20 ≈ 95 ms per 667 ms tick. Valid output from tick 0 (D6). Protocol stays wire-compatible with the OpenPI `WebsocketClientPolicy` contract.

## 5. Stage-1 code deltas

- `early_exit.py` → per-layer `(K_ℓ, V_ℓ)` export for all 36 layers (harvest a KV cache in the manual loop); the truncation and final-norm knobs move to ablation-only.
- Prompt path → remove `StateProjector`; add state discretization (quantile stats file) and text rendering; layout `[video | instruction | state]`.
- Streaming → replace 14-step `None` warm-up with front-padding, emit from tick 0.
- New: `expert.py` (spec §2), `flow.py` (conventions §2), FAST integration (tokenizer + Qwen vocab resize), joint training script with the two-loss mask.
- `realtime/` → server hosts both modules, returns action chunks; client executes 20-step segments.
- Tests (gating order): (1) dataloader ↔ streaming per-layer K/V bit-exact, including padded ticks; (2) expert mask/position unit test vs a dense reference implementation; (3) FAST round-trip (tokenize → detokenize ≈ original within quantization error); (4) overfit-one-episode end-to-end.

## 6. Evaluation

Offline, held-out episodes: 10-step denoised open-loop MSE per joint at tick-aligned samples; FAST token accuracy; replay videos (predicted vs demonstrated joints). On robot: ≥20 trials on the block task. The two comparisons that matter more than the absolute number: (1) openpi π0.5 fine-tuned on the same data, same single camera — isolates whether Qwen + temporal window + this expert clears the off-the-shelf alternative; (2) window = 1 pair ablation — the direct test of the temporal-memory bet.

## 7. Milestones

1. **M1 — Interface & data**: per-layer KV export, front-padding (both paths), state discretization, FAST tokenizer + vocab resize, dataloader with jittered sampling. Gate: equivalence + FAST round-trip tests pass.
2. **M2 — Model**: expert + flow + joint mask implemented. Gate: overfit-one-episode.
3. **M3 — Full training** (~30k steps). Gate: stable val metrics; width-768 fallback decision if overfitting.
4. **M4 — Deployment**: server integration, latency check (<150 ms/tick), robot trials vs the openpi baseline + window ablation.
5. **M5 — Extensions**: subtask annotation → hierarchical inference; wrist cameras (current-frame-only, bounded token growth); VQA co-training; real-time chunking.

## 8. Confirm before M1

Qwen2.5-VL-3B text config (expect 16 heads / 2 KV heads / head_dim 128 / hidden 2048 / 36 layers); episode count and `task` strings in `meta/info.json`; action field convention (absolute leader-arm targets?); FAST tokenizer behavior on 30×14 chunks at this normalization (token count per chunk); whether wrist camera streams exist in the dataset (for M5).