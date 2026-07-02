"""QwenVLA: backbone + FAST vocab side-car + action expert, with the single dual-loss forward.

One backbone pass per batch serves both losses (claude_plan D2, knowledge insulation):
- L_AR: cross-entropy on the FAST action tokens (+ final <|im_end|>) over the UNION vocab
  [Qwen 151936 | FAST side-car]. This is the only loss that reaches the backbone (ViT + LLM).
- L_flow: flow-matching MSE on the expert, which attends to per-layer prefix K/V exported
  DETACHED by llm_forward.forward_train — no expert gradient ever touches the backbone.

The FAST side-car (fast_embed / fast_head) replaces the plan's vocab resize: Qwen has
tie_word_embeddings=true, so resizing would mutate the tied table and un-stock-shape checkpoints.
Union ids >= V_BASE are FAST tokens.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VLMConfig
from .expert import ActionExpert, ExpertConfig
from .fast_tokens import IGNORE_INDEX, V_BASE
from .flow import flow_loss
from .llm_forward import forward_train as llm_forward_train
from .prompt_builder import ActTemplate, build_act_template


class QwenVLA(nn.Module):
    def __init__(
        self,
        backbone,                       # Qwen2_5_VLForConditionalGeneration
        processor,
        cfg: VLMConfig,
        expert_cfg: ExpertConfig,
        fast_vocab_size: int,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.expert_cfg = expert_cfg
        self.fast_vocab_size = fast_vocab_size
        self.compute_dtype = compute_dtype

        # D1 constraint (openpi asserts the same): expert attention layout must equal the backbone.
        lm_cfg = backbone.model.language_model.config
        head_dim = lm_cfg.hidden_size // lm_cfg.num_attention_heads
        checks = {
            "depth": (expert_cfg.depth, lm_cfg.num_hidden_layers),
            "num_heads": (expert_cfg.num_heads, lm_cfg.num_attention_heads),
            "num_kv_heads": (expert_cfg.num_kv_heads, lm_cfg.num_key_value_heads),
            "head_dim": (expert_cfg.head_dim, head_dim),
            "mrope_section": (tuple(expert_cfg.mrope_section),
                              tuple(lm_cfg.rope_scaling["mrope_section"])),
        }
        for name, (e, b) in checks.items():
            if e != b:
                raise ValueError(f"ExpertConfig.{name}={e} must equal backbone's {b}")
        if fast_vocab_size <= 0:
            raise ValueError("fast_vocab_size must be positive")

        self.backbone = backbone
        self.fast_embed = nn.Embedding(fast_vocab_size, lm_cfg.hidden_size)
        self.fast_head = nn.Linear(lm_cfg.hidden_size, fast_vocab_size, bias=False)
        self.expert = ActionExpert(expert_cfg)

        # New FAST embedding rows start in the pretrained table's distribution.
        with torch.no_grad():
            emb = backbone.get_input_embeddings().weight
            self.fast_embed.weight.normal_(float(emb.mean()), float(emb.std()))
            nn.init.trunc_normal_(self.fast_head.weight, std=(1.0 / lm_cfg.hidden_size) ** 0.5)

        # Constant prompt template (S_prefix is a project-wide constant).
        template: ActTemplate = build_act_template(processor, backbone, cfg)
        self.state_slice = template.state_slice
        self.S_prefix = template.S_prefix
        self.prefix_max_pos = template.prefix_max_pos
        self.register_buffer("prefix_ids", template.prefix_ids, persistent=False)
        self.register_buffer("video_mask", template.video_mask, persistent=False)
        self.register_buffer("prefix_position_ids", template.prefix_position_ids, persistent=False)
        # Expert suffix continues right after the prefix on ALL 3 mrope axes ("text mode");
        # deliberately overlaps the FAST range — the two suffixes never attend to each other.
        suffix_pos = (
            self.prefix_max_pos + 1 + torch.arange(expert_cfg.horizon, dtype=torch.long)
        ).view(1, 1, -1).expand(3, 1, -1).contiguous()
        self.register_buffer("suffix_position_ids", suffix_pos, persistent=False)
        self.template = template  # CPU copy (state_lut etc.) for dataset/policy reuse

    # ------------------------------------------------------------------ helpers
    def suffix_pos_emb(self, batch: int):
        """(cos, sin) [3, B, horizon, head_dim] via Qwen's own rotary tables."""
        lm = self.backbone.model.language_model
        ref = self.fast_head.weight[:1]  # any tensor: rotary_emb only reads device/dtype
        pos = self.suffix_position_ids.expand(-1, batch, -1)
        return lm.rotary_emb(ref, pos)

    def _embed_tail(self, tail_ids: torch.Tensor) -> torch.Tensor:
        """Union-vocab embedding for the FAST tail: base rows from Qwen, >= V_BASE from the side-car."""
        is_fast = tail_ids >= V_BASE
        base = self.backbone.get_input_embeddings()(torch.where(is_fast, torch.zeros_like(tail_ids), tail_ids))
        fast = self.fast_embed((tail_ids - V_BASE).clamp(min=0))
        return torch.where(is_fast.unsqueeze(-1), fast, base)

    def build_prefix_embeds(self, state_ids: torch.Tensor, video_embeds: torch.Tensor) -> torch.Tensor:
        """[B,56] state ids + [B, 2160, 2048] video features -> [B, S_prefix, 2048] inputs_embeds."""
        B = state_ids.shape[0]
        ids = self.prefix_ids.expand(B, -1).clone()
        ids[:, self.state_slice] = state_ids
        embeds = self.backbone.get_input_embeddings()(ids)
        mask = self.video_mask.expand(B, -1).unsqueeze(-1)
        return embeds.masked_scatter(mask, video_embeds.reshape(-1, video_embeds.shape[-1]).to(embeds.dtype))

    def param_groups(self, lr_backbone: float, lr_new: float):
        new_params = (
            list(self.fast_embed.parameters())
            + list(self.fast_head.parameters())
            + list(self.expert.parameters())
        )
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": lr_backbone},
            {"params": new_params, "lr": lr_new},
        ]

    # ------------------------------------------------------------------ training forward
    def forward_train(self, batch: Dict[str, torch.Tensor],
                      generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """Call under torch.autocast(bf16). Returns losses, metrics and the detached prefix K/V."""
        B = batch["state_ids"].shape[0]
        T_fast = batch["fast_input_ids"].shape[1]

        # 1. Vision, in-graph (D8: the ViT trains). Batched per-pair grid rows are structurally
        #    per-pair-equivalent (blocked attention); NOT PairVisionCache.encode_pair (inference_mode).
        feats = self.backbone.get_video_features(batch["pixel_values"], batch["video_grid_thw"])
        video_embeds = torch.cat(list(feats), dim=0).view(B, self.cfg.total_video_tokens, -1)

        # 2. Sequence: [constant prefix w/ fresh state ids | FAST tail].
        prefix_embeds = self.build_prefix_embeds(batch["state_ids"], video_embeds)
        tail_embeds = self._embed_tail(batch["fast_input_ids"])
        inputs_embeds = torch.cat([prefix_embeds, tail_embeds.to(prefix_embeds.dtype)], dim=1)

        tail_pos = (
            self.prefix_max_pos + 1 + torch.arange(T_fast, device=inputs_embeds.device)
        ).view(1, 1, -1).expand(3, B, -1)
        position_ids = torch.cat([self.prefix_position_ids.expand(-1, B, -1), tail_pos], dim=2)

        # 3. Backbone pass with detached per-layer prefix K/V export.
        prefix_kv, hidden = llm_forward_train(
            self.backbone, inputs_embeds, position_ids, self.S_prefix, self.compute_dtype
        )

        # 4. AR loss over the union vocab. hidden[S_prefix-1] predicts fast_0; hidden[S_prefix+n-1]
        #    predicts <|im_end|>; padded positions carry IGNORE_INDEX.
        hs = hidden[:, self.S_prefix - 1 : self.S_prefix + T_fast]
        logits = torch.cat([self.backbone.lm_head(hs), self.fast_head(hs)], dim=-1).float()
        targets = batch["ar_targets"]
        loss_ar = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX
        )
        with torch.no_grad():
            mask = targets != IGNORE_INDEX
            fast_acc = (logits.argmax(-1) == targets)[mask].float().mean()

        # 5. Flow loss on the expert (detached K/V: no gradient into the backbone).
        loss_flow, _ = flow_loss(
            self.expert, prefix_kv, self.suffix_pos_emb(B),
            batch["actions_norm"].float(), generator,
        )

        return {
            "loss": loss_ar + loss_flow,
            "loss_ar": loss_ar,
            "loss_flow": loss_flow,
            "fast_acc": fast_acc,
            "prefix_kv": prefix_kv,
        }
