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
