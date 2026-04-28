from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class STTBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a mono float32 audio array. Returns the recognized text (may be empty)."""
