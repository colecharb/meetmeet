from __future__ import annotations

from ..config import SummaryConfig
from .base import SummaryBackend


def get_backend(cfg: SummaryConfig) -> SummaryBackend:
    name = cfg.backend.lower()
    if name == "ollama":
        from .ollama_backend import OllamaBackend
        return OllamaBackend(cfg.ollama_url, cfg.model, cfg.system_prompt)
    raise ValueError(f"unknown summary backend: {cfg.backend!r}")


__all__ = ["SummaryBackend", "get_backend"]
