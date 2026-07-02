"""Output container for the streaming VLM module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch

# token_types values
TOKEN_TYPE_TEXT = 0
TOKEN_TYPE_VIDEO = 1
TOKEN_TYPE_STATE = 2


@dataclass
class VLMOutput:
    """Context-token hidden states for a future action expert to cross-attend to.

    context_tokens: [B, S, 2048] hidden states (video + instruction + state slots, in that order).
    context_mask:   [B, S] 1 where a token is valid (all ones in v1; no padding).
    token_types:    [B, S] one of TOKEN_TYPE_{TEXT,VIDEO,STATE}.
    layer_index:    number of LLM decoder layers actually run (== cfg.num_llm_layers_to_run).
    metadata:       free-form diagnostics (tolerances, shapes, timings, ...).
    """

    context_tokens: torch.Tensor
    context_mask: torch.Tensor
    token_types: torch.Tensor
    layer_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
