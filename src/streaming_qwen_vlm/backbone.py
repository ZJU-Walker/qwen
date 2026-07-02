"""Load the Qwen2.5-VL-3B backbone + processor, with a FlashAttention-2 -> SDPA fallback."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .config import VLMConfig

logger = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {list(_DTYPES)}")
    return _DTYPES[name]


def load_backbone(cfg: VLMConfig) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Return (model, processor) on cfg.device in cfg.dtype, eval mode.

    Tries cfg.attn_impl first; on a known attention-backend failure falls back to "sdpa".
    """
    dtype = resolve_dtype(cfg.dtype)

    def _load(attn_impl: str) -> Qwen2_5_VLForConditionalGeneration:
        # transformers>=4.57 renamed torch_dtype -> dtype.
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_id,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

    try:
        model = _load(cfg.attn_impl)
    except (ImportError, ValueError, RuntimeError) as exc:
        if cfg.attn_impl == "sdpa":
            raise
        logger.warning(
            "attn_implementation=%r failed (%s: %s); falling back to 'sdpa'.",
            cfg.attn_impl, type(exc).__name__, exc,
        )
        model = _load("sdpa")

    model = model.to(cfg.device).eval()

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    return model, processor
