from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class SummaryBackend(ABC):
    @abstractmethod
    def summarize(self, transcript: str, title: str) -> Iterator[str]:
        """Yield summary text incrementally as it is produced."""
