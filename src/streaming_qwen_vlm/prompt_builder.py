"""Build the chat-template prompt ONCE and cache it.

The whole rolling window is a SINGLE logical video placeholder. We run the processor on a dummy zeros
video of ``window_frames`` frames at the fixed resolution (grid pinned via min==max pixels), which
expands the video placeholder into exactly ``total_video_tokens`` ``video_token_id`` slots. Everything
returned here is fixed for the lifetime of the model; changing ``cfg.instruction`` requires a rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import VLMConfig, VIDEO_TOKEN_ID
from .state_text import N_STATE_TOKENS, STATE_DIMS, bins_to_ids, build_state_lut

# The generation-prompt tail Qwen's chat template appends. build_act_template splits the rendered
# chat text on this constant so the state ids can be spliced INSIDE the user turn.
_GEN_TAIL = "<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class PromptTemplate:
    input_ids: torch.Tensor          # [1, S_template]
    attention_mask: torch.Tensor     # [1, S_template]
    video_grid_thw: torch.Tensor     # [[num_pairs, grid_h, grid_w]]
    second_per_grid_ts: torch.Tensor # [num_videos]
    video_mask: torch.Tensor         # [1, S_template] bool, True at video_token_id slots
    text_mask: torch.Tensor          # [1, S_template] bool, True at non-video (instruction/scaffold) slots
    instr_len: int                   # number of non-video template tokens


def build_prompt(processor, cfg: VLMConfig) -> PromptTemplate:
    h, w = cfg.fixed_resolution
    dummy_video = np.zeros((cfg.window_frames, h, w, 3), dtype=np.uint8)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": cfg.instruction},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        videos=[dummy_video],
        fps=cfg.fps,
        min_pixels=h * w,
        max_pixels=h * w,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    video_grid_thw = inputs["video_grid_thw"]

    n_video = int((input_ids == VIDEO_TOKEN_ID).sum().item())
    if n_video != cfg.total_video_tokens:
        raise ValueError(
            f"Template expanded to {n_video} video tokens, expected {cfg.total_video_tokens}. "
            f"video_grid_thw={video_grid_thw.tolist()}"
        )

    video_mask = input_ids == VIDEO_TOKEN_ID
    text_mask = ~video_mask
    second = inputs.get("second_per_grid_ts", [cfg.second_per_grid_ts] * len(video_grid_thw))
    second_per_grid_ts = torch.as_tensor(second, dtype=torch.float32)

    return PromptTemplate(
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"],
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        video_mask=video_mask,
        text_mask=text_mask,
        instr_len=int(text_mask.sum().item()),
    )


# --------------------------------------------------------------------------------------------- #
# Stage 2: the VLA prompt template.                                                              #
# --------------------------------------------------------------------------------------------- #


@dataclass
class ActTemplate:
    """Constant VLA prefix: [scaffold | 2160 video | instruction | "\\nState:" | 56 state ids | tail].

    S_prefix is a single project-wide constant (state text is constant-length), so masks, mrope
    positions and the expert's key layout are all static. Per sample/tick only ``state_slice`` is
    overwritten with fresh bin ids.
    """

    prefix_ids: torch.Tensor            # [1, S_prefix] int64 (state slots hold bin-128 placeholders)
    video_mask: torch.Tensor            # [1, S_prefix] bool, True at the 2160 video slots
    state_slice: slice                  # constant offsets of the 56 state ids
    S_prefix: int
    video_grid_thw: torch.Tensor        # [[num_pairs, grid_h, grid_w]]
    second_per_grid_ts: torch.Tensor    # [1] float32
    prefix_position_ids: torch.Tensor   # [3, 1, S_prefix] mrope positions (computed once)
    prefix_max_pos: int                 # int(prefix_position_ids.max()); FAST + expert suffixes start at +1
    state_lut: torch.Tensor             # [256, 4] token-id LUT for bins_to_ids


def build_act_template(processor, model, cfg: VLMConfig) -> ActTemplate:
    """Build the constant VLA prefix. ``model`` is only used for ``get_rope_index``.

    All tensors are on CPU; consumers move them to the right device.
    """
    h, w = cfg.fixed_resolution
    tokenizer = processor.tokenizer
    state_lut = build_state_lut(tokenizer)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": cfg.instruction},
            ],
        }
    ]
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if full_text.count(_GEN_TAIL) != 1 or not full_text.endswith(_GEN_TAIL):
        raise AssertionError(
            f"chat template does not end with the expected generation tail {_GEN_TAIL!r}; "
            f"got: ...{full_text[-80:]!r}"
        )

    # Head: everything up to (excluding) the tail, plus the state prefix text. Runs through the
    # full processor so the video placeholder expands to 2160 video ids.
    head_text = full_text[: -len(_GEN_TAIL)] + cfg.state_prefix_text
    dummy_video = np.zeros((cfg.window_frames, h, w, 3), dtype=np.uint8)
    inputs = processor(
        text=[head_text],
        videos=[dummy_video],
        fps=cfg.fps,
        min_pixels=h * w,
        max_pixels=h * w,
        return_tensors="pt",
    )
    head_ids = inputs["input_ids"]                     # [1, S_head]
    video_grid_thw = inputs["video_grid_thw"]

    n_video = int((head_ids == VIDEO_TOKEN_ID).sum().item())
    if n_video != cfg.total_video_tokens:
        raise ValueError(
            f"ActTemplate expanded to {n_video} video tokens, expected {cfg.total_video_tokens}; "
            f"video_grid_thw={video_grid_thw.tolist()}"
        )

    # State segment: 56 ids, placeholder bin 128 everywhere (overwritten per sample).
    state_ids = bins_to_ids([128] * STATE_DIMS, state_lut).unsqueeze(0)  # [1, 56]

    # Tail: the generation prompt, tokenized as plain text.
    tail_ids = torch.tensor([tokenizer.encode(_GEN_TAIL, add_special_tokens=False)], dtype=torch.long)

    prefix_ids = torch.cat([head_ids, state_ids, tail_ids], dim=1)
    S_head = head_ids.shape[1]
    S_prefix = prefix_ids.shape[1]
    state_slice = slice(S_head, S_head + N_STATE_TOKENS)

    second = inputs.get("second_per_grid_ts", [cfg.second_per_grid_ts] * len(video_grid_thw))
    second_per_grid_ts = torch.as_tensor(second, dtype=torch.float32)

    # mrope positions for the whole constant prefix (state text is ordinary in-sequence text —
    # no manual position extension needed anywhere anymore).
    prefix_position_ids, _ = model.model.get_rope_index(
        prefix_ids,
        image_grid_thw=None,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=torch.ones_like(prefix_ids),
    )
    prefix_position_ids = prefix_position_ids.cpu()

    return ActTemplate(
        prefix_ids=prefix_ids,
        video_mask=prefix_ids == VIDEO_TOKEN_ID,
        state_slice=state_slice,
        S_prefix=S_prefix,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        prefix_position_ids=prefix_position_ids,
        prefix_max_pos=int(prefix_position_ids.max().item()),
        state_lut=state_lut,
    )
