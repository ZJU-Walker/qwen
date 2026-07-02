"""Configuration for the streaming Qwen2.5-VL VLM feature module.

All numbers here are verified against the live Qwen2.5-VL-3B-Instruct config.json and the
transformers 4.57.6 source. See claude_plan.md for the source-line references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# --- Fixed Qwen2.5-VL-3B constants (do not change without re-verifying against the checkpoint) ---
D = 2048                  # LLM hidden_size
NUM_LLM_LAYERS = 36       # text-model num_hidden_layers
PATCH = 14                # vision patch_size
MERGE = 2                 # spatial_merge_size
TEMPORAL_PATCH = 2        # temporal_patch_size (frames collapsed per temporal patch)
TOKENS_PER_SECOND = 2     # vision tokens_per_second (mrope temporal scaling)
VIDEO_TOKEN_ID = 151656
IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652
PIXEL_FACTOR = PATCH * MERGE  # 28: smart_resize rounds H,W to a multiple of this

DEFAULT_INSTRUCTION = "Watch the human, then put the block they indicated onto the plate."


@dataclass
class VLMConfig:
    """Knobs for the streaming VLM module plus derived grid math.

    The vision-token layout MUST be identical for every encoded pair (the pair cache concatenates
    pre-computed embeddings), so resolution is pinned via min_pixels==max_pixels in preprocessing.
    """

    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    dtype: str = "bfloat16"
    attn_impl: str = "flash_attention_2"
    device: str = "cuda"

    # Streaming / temporal layout
    fps: int = 3
    window_seconds: int = 10
    frames_per_pair: int = 2
    num_pairs: int = 15
    fixed_resolution: Tuple[int, int] = (336, 336)  # (H, W); each a multiple of PIXEL_FACTOR
    min_pixels: Optional[int] = None  # filled in __post_init__ to pin the grid
    max_pixels: Optional[int] = None

    # Robot state encoding.
    # Stage 2 (VLA): state is discretized into the prompt as text (state_bins / state_prefix_text)
    # and num_state_tokens defaults to 0 — the Stage-1 StateProjector token is retired from the
    # main path (kept importable for ablations by setting num_state_tokens > 0).
    state_dim: int = 14
    num_state_tokens: int = 0
    state_hidden_dim: int = 256
    state_seed: int = 0
    state_bins: int = 256
    state_prefix_text: str = "\nState:"

    # Stage-2 action head knobs (used by fast_tokens / policy; expert dims live in ExpertConfig).
    # max_fast_tokens: measured over ALL 8868 train chunks (fast_tokens --measure --stride 1):
    # max 147, p99 134 -> 192 leaves ~30% headroom incl. held-out episodes.
    max_fast_tokens: int = 192
    num_denoise_steps: int = 10

    # Early-exit / compute knobs
    num_llm_layers_to_run: int = NUM_LLM_LAYERS
    early_exit_norm: str = "final_norm"  # "final_norm" | "none"
    feature_layer: str = "last_ran"      # debug only

    instruction: str = DEFAULT_INSTRUCTION

    # --- derived (populated in __post_init__) ---
    window_frames: int = field(init=False)
    second_per_grid_ts: float = field(init=False)
    grid_h: int = field(init=False)
    grid_w: int = field(init=False)
    tokens_per_pair: int = field(init=False)
    total_video_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        h, w = self.fixed_resolution
        if h % PIXEL_FACTOR != 0 or w % PIXEL_FACTOR != 0:
            raise ValueError(
                f"fixed_resolution {self.fixed_resolution} must be a multiple of {PIXEL_FACTOR} "
                f"(patch*merge) so smart_resize is a no-op and the grid is deterministic."
            )
        if not (1 <= self.num_llm_layers_to_run <= NUM_LLM_LAYERS):
            raise ValueError(
                f"num_llm_layers_to_run must be in [1, {NUM_LLM_LAYERS}], got {self.num_llm_layers_to_run}"
            )
        if self.early_exit_norm not in ("final_norm", "none"):
            raise ValueError(f"early_exit_norm must be 'final_norm' or 'none', got {self.early_exit_norm!r}")

        self.window_frames = self.num_pairs * self.frames_per_pair
        # window_seconds is derived from the window so num_pairs can be changed freely (e.g. a shorter
        # 10-frame history). For the default (num_pairs=15, frames_per_pair=2, fps=3) this is still 10s.
        self.window_seconds = self.window_frames / self.fps

        # second_per_grid_ts = temporal_patch_size / fps (processor convention). With TEMPORAL_PATCH==
        # frames_per_pair==2 this equals frames_per_pair/fps.
        self.second_per_grid_ts = self.frames_per_pair / self.fps

        self.grid_h = h // PATCH
        self.grid_w = w // PATCH
        # spatial_merge collapses a 2x2 block of patches into one token.
        assert (self.grid_h * self.grid_w) % (MERGE * MERGE) == 0
        self.tokens_per_pair = (self.grid_h * self.grid_w) // (MERGE * MERGE)
        self.total_video_tokens = self.tokens_per_pair * self.num_pairs

        # Pin the processor grid: with min==max==H*W, smart_resize cannot rescale.
        if self.min_pixels is None:
            self.min_pixels = h * w
        if self.max_pixels is None:
            self.max_pixels = h * w

    # --- grid_thw views (T, H_patches, W_patches) ---
    @property
    def video_grid_thw(self) -> Tuple[int, int, int]:
        """The whole rolling window as ONE logical video row: (num_pairs, grid_h, grid_w)."""
        return (self.num_pairs, self.grid_h, self.grid_w)

    @property
    def pair_grid_thw(self) -> Tuple[int, int, int]:
        """A single encoded pair (2 frames -> 1 temporal patch): (1, grid_h, grid_w)."""
        return (1, self.grid_h, self.grid_w)
