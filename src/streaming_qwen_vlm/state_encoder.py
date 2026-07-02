"""Deterministic robot-state -> context-token projector.

v1 is a fixed, untrained MLP whose weights are seeded so repeated runs/tests are reproducible. It maps
a state vector to num_state_tokens hidden vectors of width 2048 that are appended to the LLM input.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import D, VLMConfig


class StateProjector(nn.Module):
    def __init__(self, cfg: VLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_state_tokens = cfg.num_state_tokens
        out_dim = cfg.num_state_tokens * D
        # Deterministic init: build the layers under a forked, seeded RNG so global state is untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.state_seed)
            self.net = nn.Sequential(
                nn.Linear(cfg.state_dim, cfg.state_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.state_hidden_dim, out_dim),
            )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [B, state_dim] -> [B, num_state_tokens, 2048] in the projector's dtype/device."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        w = next(self.net.parameters())
        state = state.to(device=w.device, dtype=w.dtype)
        h = self.net(state)
        return h.view(state.shape[0], self.num_state_tokens, D)
