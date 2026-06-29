"""Environment-driven configuration.

All secrets live in ``API.env`` (git-ignored); see ``.env.example`` for the
full list of keys. ``load_settings()`` reads that file once and returns a frozen
``Settings`` object the rest of the package can depend on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_GENERATOR_MODEL = "meta-llama/llama-3.2-3b-instruct"
DEFAULT_JUDGE_MODEL = "openai/gpt-5-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for a run."""

    openrouter_api_key: str | None
    cohere_api_key: str | None
    huggingface_api_key: str | None
    phoenix_api_key: str | None
    weaviate_url: str | None
    weaviate_api_key: str | None
    workspace_id: str | None
    base_url: str = DEFAULT_BASE_URL
    generator_model: str = DEFAULT_GENERATOR_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL

    def require(self, *names: str) -> None:
        """Raise if any of the named settings are missing — fail fast and loud."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"Missing required settings: {', '.join(missing)}. "
                "Copy .env.example to API.env and fill in your keys."
            )


def load_settings(env_file: str | os.PathLike = "API.env") -> Settings:
    """Load settings from ``env_file`` (if present) and the process environment."""
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)

    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY"),
        phoenix_api_key=os.getenv("PHOENIX_API_KEY"),
        weaviate_url=os.getenv("WEAVIATE_URL"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
        workspace_id=os.getenv("WORKSPACE_ID"),
    )
