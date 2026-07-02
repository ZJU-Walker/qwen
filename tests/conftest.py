"""Shared fixtures: load the 3B model once per session; deterministic synthetic frames."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from streaming_qwen_vlm import VLMConfig, load_backbone


@pytest.fixture(scope="session")
def cfg():
    return VLMConfig()


@pytest.fixture(scope="session")
def backbone(cfg):
    model, processor = load_backbone(cfg)
    return model, processor


@pytest.fixture(scope="session")
def model(backbone):
    return backbone[0]


@pytest.fixture(scope="session")
def processor(backbone):
    return backbone[1]


def make_frames(n: int, seed: int = 0, size=(480, 640)):
    """Deterministic random uint8 RGB frames [n, H, W, 3]."""
    rng = np.random.default_rng(seed)
    h, w = size
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def tagged_frames(n: int, size=(336, 336)):
    """Frames where frame i is a constant plane of value i (sentinel-identifiable)."""
    h, w = size
    return [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n)]


@pytest.fixture(scope="session")
def synthetic_window(cfg):
    """A full window of deterministic frames at the fixed resolution."""
    return make_frames(cfg.window_frames, seed=1234, size=cfg.fixed_resolution)
