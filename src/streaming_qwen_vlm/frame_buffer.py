"""Rolling 30-frame window that emits the newest un-encoded 2-frame pair.

In the streaming model we usually hand the model exactly two new frames per step, so this buffer is a
convenience for callers that push one frame at a time. It tracks how many buffered frames have already
been consumed into pairs and yields the next complete pair when >= frames_per_pair new frames exist.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class RollingFrameBuffer:
    def __init__(self, window_frames: int, frames_per_pair: int = 2) -> None:
        self.window_frames = window_frames
        self.frames_per_pair = frames_per_pair
        self._buf: Deque[np.ndarray] = deque(maxlen=window_frames)
        self._unconsumed = 0  # frames added but not yet returned as part of a pair

    def add_frame(self, rgb: np.ndarray) -> None:
        self._buf.append(np.asarray(rgb))
        self._unconsumed = min(self._unconsumed + 1, self.window_frames)

    def pop_new_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the next un-encoded pair (oldest-unconsumed two frames) or None if < 2 are ready."""
        if self._unconsumed < self.frames_per_pair:
            return None
        frames = list(self._buf)
        # The pair starts at the first unconsumed frame.
        start = len(frames) - self._unconsumed
        a, b = frames[start], frames[start + 1]
        self._unconsumed -= self.frames_per_pair
        return a, b

    def __len__(self) -> int:
        return len(self._buf)
