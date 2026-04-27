from __future__ import annotations

from ..config import STTConfig
from .base import STTBackend


def get_backend(cfg: STTConfig) -> STTBackend:
    name = cfg.backend.lower()
    if name == "mlx":
        from .mlx_whisper_backend import MLXWhisperBackend
        return MLXWhisperBackend(cfg.model)
    if name == "faster":
        from .faster_whisper_backend import FasterWhisperBackend
        return FasterWhisperBackend(cfg.model)
    raise ValueError(f"unknown stt backend: {cfg.backend!r}")


__all__ = ["STTBackend", "get_backend"]
