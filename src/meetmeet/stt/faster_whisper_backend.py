from __future__ import annotations

import numpy as np

from .base import STTBackend


class FasterWhisperBackend(STTBackend):
    def __init__(self, model: str = "large-v3-turbo"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed. Install with: pip install meetmeet[faster]"
            ) from e
        self._model = WhisperModel(model, compute_type="int8")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError(f"faster-whisper expects 16kHz audio, got {sample_rate}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        segments, _ = self._model.transcribe(audio, language="en", beam_size=1)
        return " ".join(seg.text.strip() for seg in segments if seg.text).strip()
