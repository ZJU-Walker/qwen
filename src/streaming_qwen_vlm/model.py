"""StreamingQwenVLM: the streaming Qwen2.5-VL-3B context-feature module.

Per step it: encodes the newest 2-frame pair (cached), reconstructs the 30-frame window feature from
the pair cache, injects it into the LLM input via masked_scatter (vision encoder NOT re-run), appends
robot-state tokens, runs a configurable number of LLM layers (real early exit), and returns the
context-token hidden states grouped as [video | instruction | state].
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

from .backbone import load_backbone, resolve_dtype
from .config import VIDEO_TOKEN_ID, VLMConfig
from .early_exit import run_language_early_exit
from .frame_buffer import RollingFrameBuffer
from .outputs import TOKEN_TYPE_STATE, TOKEN_TYPE_TEXT, TOKEN_TYPE_VIDEO, VLMOutput
from .preprocess import pair_to_pixel_values, prepare_frames
from .prompt_builder import PromptTemplate, build_prompt
from .state_encoder import StateProjector
from .vision_cache import PairVisionCache

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence]


class StreamingQwenVLM:
    def __init__(self, cfg: VLMConfig, model=None, processor=None) -> None:
        self.cfg = cfg
        if model is None or processor is None:
            model, processor = load_backbone(cfg)
        self.model = model
        self.processor = processor
        self.device = model.device
        self.dtype = resolve_dtype(cfg.dtype)

        self.cache = PairVisionCache(cfg.num_pairs, cfg.tokens_per_pair)
        self.frame_buffer = RollingFrameBuffer(cfg.window_frames, cfg.frames_per_pair)
        self.state_proj = StateProjector(cfg).to(self.device, self.dtype)

        self.template: PromptTemplate = build_prompt(processor, cfg)
        self._input_ids = self.template.input_ids.to(self.device)
        self._video_mask = self.template.video_mask.to(self.device)
        self._text_mask = self.template.text_mask.to(self.device)
        self._attn_template = self.template.attention_mask.to(self.device)
        self._S_template = self._input_ids.shape[1]

        # Base mrope positions for the fixed template (computed once).
        base_pos, _ = self.model.model.get_rope_index(
            self._input_ids,
            image_grid_thw=None,
            video_grid_thw=self.template.video_grid_thw.to(self.device),
            second_per_grid_ts=self.template.second_per_grid_ts.to(self.device),
            attention_mask=self._attn_template,
        )
        self._base_position_ids = base_pos  # [3, 1, S_template]

        # Precompute the extended (template + state) positions and attention mask.
        n_state = cfg.num_state_tokens
        if n_state > 0:
            max_pos = int(self._base_position_ids.max())
            ext = (torch.arange(1, n_state + 1, device=self.device) + max_pos)
            ext = ext.view(1, 1, -1).expand(3, 1, -1)
            self._position_ids = torch.cat([self._base_position_ids, ext], dim=2)
            self._attn_full = torch.cat(
                [self._attn_template, torch.ones(1, n_state, device=self.device, dtype=self._attn_template.dtype)],
                dim=1,
            )
        else:
            self._position_ids = self._base_position_ids
            self._attn_full = self._attn_template

        # Static token_types / context_mask (layout is fixed: video | instruction | state).
        n_video = cfg.total_video_tokens
        instr_len = self.template.instr_len
        types = (
            [TOKEN_TYPE_VIDEO] * n_video
            + [TOKEN_TYPE_TEXT] * instr_len
            + [TOKEN_TYPE_STATE] * n_state
        )
        self._token_types = torch.tensor(types, device=self.device, dtype=torch.long).unsqueeze(0)
        self._S_context = n_video + instr_len + n_state

    # --- internal ---
    def _encode_and_push_pair(self, new_two_frames: ArrayLike) -> None:
        if isinstance(new_two_frames, torch.Tensor):
            new_two_frames = new_two_frames.cpu().numpy()
        frames = list(new_two_frames)
        if len(frames) != self.cfg.frames_per_pair:
            raise ValueError(f"Expected {self.cfg.frames_per_pair} frames, got {len(frames)}")
        for f in frames:
            self.frame_buffer.add_frame(f)
        prepared = prepare_frames(frames, self.cfg)  # [2, H, W, 3] uint8
        pair_inputs = pair_to_pixel_values(self.processor, prepared, self.cfg)
        z = self.cache.encode_pair(
            self.model,
            pair_inputs["pixel_values_videos"],
            pair_inputs["video_grid_thw"],
        )
        self.cache.push(z.to(self.dtype))

    @torch.inference_mode()
    def step(self, new_two_frames: ArrayLike, robot_state: ArrayLike) -> Optional[VLMOutput]:
        cfg = self.cfg
        self._encode_and_push_pair(new_two_frames)
        if not self.cache.is_full():
            return None  # warm-up: window not yet full

        # 1. Reconstruct the window feature from cached pair embeddings.
        video_embeds = self.cache.concat().to(self.device, self.dtype)  # [total_video_tokens, 2048]

        # 2. Build inputs_embeds and scatter the video features into the video_token_id slots.
        inputs_embeds = self.model.get_input_embeddings()(self._input_ids)  # [1, S_t, 2048]
        scatter_mask = self._video_mask.unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(scatter_mask, video_embeds.reshape(-1))

        # 3. Append robot-state tokens.
        n_state = cfg.num_state_tokens
        if n_state > 0:
            state = robot_state
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(np.asarray(state, dtype=np.float32))
            state_tok = self.state_proj(state).to(self.device, self.dtype)  # [1, n_state, 2048]
            inputs_embeds = torch.cat([inputs_embeds, state_tok], dim=1)

        # 4. Run real early-exit LLM.
        hs = run_language_early_exit(
            self.model,
            inputs_embeds=inputs_embeds,
            attention_mask=self._attn_full,
            position_ids=self._position_ids,
            num_layers=cfg.num_llm_layers_to_run,
            early_exit_norm=cfg.early_exit_norm,
        )  # [1, S_t + n_state, 2048]

        # 5. Slice into video | instruction | state and concatenate in that order.
        hs_template = hs[:, : self._S_template, :]
        video_hs = hs_template[self._video_mask]   # [total_video_tokens, 2048]
        instr_hs = hs_template[self._text_mask]     # [instr_len, 2048]
        parts = [video_hs, instr_hs]
        if n_state > 0:
            state_hs = hs[:, self._S_template :, :].reshape(n_state, -1)
            parts.append(state_hs)
        context_tokens = torch.cat(parts, dim=0).unsqueeze(0)  # [1, S_context, 2048]

        context_mask = torch.ones(
            1, self._S_context, device=self.device, dtype=torch.long
        )
        return VLMOutput(
            context_tokens=context_tokens,
            context_mask=context_mask,
            token_types=self._token_types,
            layer_index=cfg.num_llm_layers_to_run,
            metadata={
                "num_video_tokens": cfg.total_video_tokens,
                "instr_len": self.template.instr_len,
                "num_state_tokens": n_state,
                "early_exit_norm": cfg.early_exit_norm,
                "S_context": self._S_context,
            },
        )
