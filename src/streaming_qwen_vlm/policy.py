"""ActPolicy: inference wrapper — (2 frames, raw state) -> [horizon, 14] action chunk.

Per tick: encode the newest pair into the rolling cache (front-padded until full — valid output
from tick 0), splice the discretized state into the constant prompt, run the 36-layer prefill once
(per-layer K/V export), then 10 expert Euler steps reusing that PrefixKV, and unnormalize.

Loads a ``step_XXXXXX`` checkpoint written by training/checkpoints.py. The FAST side-car is not
loaded — no tokens are decoded at inference.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .config import VLMConfig
from .expert import ActionExpert, ExpertConfig
from .flow import sample_actions
from .llm_forward import forward_prefill
from .normalize import NormStats, discretize, load_stats, unnormalize
from .preprocess import pair_to_pixel_values, prepare_frames
from .prompt_builder import build_act_template
from .state_text import bins_to_ids
from .timing import cuda_timer
from .vision_cache import PairVisionCache

logger = logging.getLogger(__name__)


def _load_backbone_from_dir(path: str, attn_impl: str = "flash_attention_2"):
    try:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation=attn_impl)
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.warning("attn_implementation=%r failed (%s); falling back to sdpa", attn_impl, exc)
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa")


class ActPolicy:
    def __init__(self, checkpoint_dir: str, device: str = "cuda",
                 num_denoise_steps: int | None = None, use_ema: bool = True) -> None:
        with open(os.path.join(checkpoint_dir, "meta.json")) as f:
            meta = json.load(f)
        vlm_kwargs = dict(meta["vlm_config"])
        vlm_kwargs["fixed_resolution"] = tuple(vlm_kwargs["fixed_resolution"])
        self.cfg = VLMConfig(**vlm_kwargs)
        exp_kwargs = dict(meta["expert_config"])
        exp_kwargs["mrope_section"] = tuple(exp_kwargs["mrope_section"])
        self.expert_cfg = ExpertConfig(**exp_kwargs)
        self.num_denoise_steps = num_denoise_steps or self.cfg.num_denoise_steps
        self.device = torch.device(device)

        bdir = os.path.join(checkpoint_dir, "backbone")
        self.model = _load_backbone_from_dir(bdir).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(bdir)

        self.expert = ActionExpert(self.expert_cfg)
        if use_ema:
            sd = load_file(os.path.join(checkpoint_dir, "expert_ema.safetensors"))
        else:
            heads = load_file(os.path.join(checkpoint_dir, "vla_heads.safetensors"))
            sd = {k[len("expert."):]: v for k, v in heads.items() if k.startswith("expert.")}
        self.expert.load_state_dict(sd, strict=True)
        self.expert = self.expert.to(self.device, torch.bfloat16).eval()

        stats = load_stats(os.path.join(checkpoint_dir, "norm_stats.json"))
        self.action_stats: NormStats = stats["action"]
        self.state_stats: NormStats = stats["state"]

        tmpl = build_act_template(self.processor, self.model, self.cfg)
        if tmpl.S_prefix != meta["S_prefix"]:
            raise RuntimeError(
                f"rebuilt template S_prefix={tmpl.S_prefix} != checkpoint {meta['S_prefix']} — "
                f"instruction/tokenizer mismatch")
        self.template = tmpl
        self.prefix_ids = tmpl.prefix_ids.to(self.device)
        self.video_mask = tmpl.video_mask.to(self.device)
        self.prefix_position_ids = tmpl.prefix_position_ids.to(self.device)

        lm = self.model.model.language_model
        suffix_pos = (
            tmpl.prefix_max_pos + 1 + torch.arange(self.expert_cfg.horizon, device=self.device)
        ).view(1, 1, -1).expand(3, 1, -1)
        ref = torch.zeros(1, device=self.device, dtype=torch.bfloat16)
        self.suffix_pos_emb = lm.rotary_emb(ref, suffix_pos)

        self.cache = PairVisionCache(self.cfg.num_pairs, self.cfg.tokens_per_pair)

    def reset(self) -> None:
        self.cache = PairVisionCache(self.cfg.num_pairs, self.cfg.tokens_per_pair)

    @torch.inference_mode()
    def act(self, two_frames: np.ndarray, state_raw: np.ndarray) -> Dict:
        """two_frames uint8 [2, H, W, 3] RGB; state_raw float32 [14] -> actions float32 [horizon, 14]."""
        cfg = self.cfg
        with cuda_timer() as t_vis:
            prepared = prepare_frames(list(np.asarray(two_frames)), cfg)
            pair = pair_to_pixel_values(self.processor, prepared, cfg)
            z = self.cache.encode_pair(self.model, pair["pixel_values_videos"], pair["video_grid_thw"])
            self.cache.push(z.to(torch.bfloat16))
            video_embeds = self.cache.concat(pad_to_full=True)  # [2160, 2048], front-padded

        with cuda_timer() as t_prefill:
            bins = discretize(np.asarray(state_raw, dtype=np.float32), self.state_stats, cfg.state_bins)
            ids = self.prefix_ids.clone()
            ids[0, self.template.state_slice] = bins_to_ids(bins, self.template.state_lut).to(self.device)
            embeds = self.model.get_input_embeddings()(ids)
            embeds = embeds.masked_scatter(
                self.video_mask.unsqueeze(-1), video_embeds.reshape(-1).to(embeds.dtype))
            prefix_kv = forward_prefill(self.model, embeds, self.prefix_position_ids)

        with cuda_timer() as t_denoise:
            chunk = sample_actions(
                self.expert, prefix_kv, self.suffix_pos_emb,
                self.expert_cfg.horizon, self.expert_cfg.action_dim, self.num_denoise_steps,
            )
        actions = unnormalize(chunk[0].float().cpu().numpy(), self.action_stats)

        return {
            "actions": actions,                       # float32 [horizon, 14] raw joint targets
            "num_pairs": len(self.cache),
            "timings": {"vision_ms": t_vis(), "prefill_ms": t_prefill(), "denoise_ms": t_denoise()},
        }
