"""Lightweight energy-based voice activity detection for barge-in.

Good enough to detect "the operator started talking" within one 20 ms block.
Upgrade path: webrtcvad behind the same interface.
"""

from __future__ import annotations

import numpy as np


class EnergyVAD:
    def __init__(
        self,
        *,
        threshold_db: float = -35.0,
        attack_blocks: int = 2,
        release_blocks: int = 25,  # ~0.5 s of silence to release
    ) -> None:
        self.threshold_db = threshold_db
        self.attack_blocks = attack_blocks
        self.release_blocks = release_blocks
        self._above = 0
        self._below = 0
        self.active = False

    def feed(self, pcm: bytes) -> bool:
        """Feed one capture block; returns current speech-active state."""
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        if samples.size == 0:
            return self.active
        rms = float(np.sqrt(np.mean(samples**2)))
        db = -120.0 if rms < 1e-9 else 20.0 * np.log10(rms / 32768.0)

        if db >= self.threshold_db:
            self._above += 1
            self._below = 0
            if self._above >= self.attack_blocks:
                self.active = True
        else:
            self._below += 1
            self._above = 0
            if self._below >= self.release_blocks:
                self.active = False
        return self.active
