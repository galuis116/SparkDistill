"""OpenAI-compatible chat API for evaluating distilled models."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ModelInterface:
    def __init__(self, endpoint: str, model: str, api_key: str = ""):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key

    @classmethod
    def from_config(cls, config: Any) -> ModelInterface:
        return cls(
            endpoint=config.model_endpoint,
            model=config.model_name,
            api_key=config.api_key,
        )

    def generate(
        self,
        prompt: dict[str, str],
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        messages: list[dict[str, str]] = []
        if prompt.get("system"):
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt["user"]})

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{self.endpoint}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"model API error HTTP {e.code}: {detail}") from e
        return data["choices"][0]["message"]["content"]
