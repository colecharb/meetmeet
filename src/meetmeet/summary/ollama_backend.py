from __future__ import annotations

import json
from typing import Iterator

import httpx

from .base import SummaryBackend


class OllamaBackend(SummaryBackend):
    def __init__(self, url: str, model: str, system_prompt: str):
        self.url = url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt

    def summarize(self, transcript: str, title: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "stream": True,
            "think": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Meeting title: {title}\n\nTranscript:\n{transcript}",
                },
            ],
        }
        with httpx.stream(
            "POST", f"{self.url}/api/chat", json=payload, timeout=None
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = (obj.get("message") or {}).get("content", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    break

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def has_model(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=2.0)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return False
        names = {m.get("name", "") for m in data.get("models", [])}
        return self.model in names or any(n.startswith(f"{self.model}:") for n in names)
