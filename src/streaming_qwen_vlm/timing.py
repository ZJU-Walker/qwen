"""CUDA-event timing + peak-memory helpers for the benchmark."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Tuple

import torch


@contextmanager
def cuda_timer():
    """Context manager yielding a callable that returns elapsed GPU time in ms.

    Usage:
        with cuda_timer() as elapsed:
            ... gpu work ...
        ms = elapsed()
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    result = {"ms": None}
    start.record()
    try:
        yield lambda: result["ms"]
    finally:
        end.record()
        torch.cuda.synchronize()
        result["ms"] = start.elapsed_time(end)


def reset_peak_memory(device=None) -> None:
    torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device=None) -> Tuple[float, float]:
    """Return (peak_allocated_MB, peak_reserved_MB)."""
    alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return alloc, reserved
