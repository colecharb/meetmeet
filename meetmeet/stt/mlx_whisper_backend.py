from __future__ import annotations

import numpy as np

from .base import STTBackend


class MLXWhisperBackend(STTBackend):
    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo"):
        self.model = model

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import mlx_whisper

        if sample_rate != 16000:
            raise ValueError(f"mlx-whisper expects 16kHz audio, got {sample_rate}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            language="en",
            fp16=True,
        )
        return (result.get("text") or "").strip()
