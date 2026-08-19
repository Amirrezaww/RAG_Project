"""Provider-agnostic LLM client.

A thin wrapper over any OpenAI-compatible chat-completions endpoint
(OpenRouter, Together, OpenAI, ...). Keeping it dependency-light (``requests``)
makes the call surface obvious and easy to trace.
"""
from __future__ import annotations

import json
from typing import Any

import requests

from .config import DEFAULT_BASE_URL, DEFAULT_GENERATOR_MODEL

Message = dict[str, str]


class LLMClient:
    """Minimal chat client for OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_GENERATOR_MODEL,
        timeout: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError("An API key is required to create an LLMClient.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        role: str = "user",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 500,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Single-turn generation; returns the assistant's text content."""
        messages = [{"role": role, "content": prompt}]
        return self.generate_messages(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model=model,
            **kwargs,
        )

    def generate_messages(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 500,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Multi-turn generation from a chat history; returns text content."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(f"LLM API error ({response.status_code}): {response.text}")

        try:
            data = response.json()
            return data["choices"][-1]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unexpected LLM response: {response.text}") from exc

    # -- internals ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter-specific headers; harmless for other providers.
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Multi-Strategy RAG",
        }
