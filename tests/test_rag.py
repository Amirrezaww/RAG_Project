"""Unit tests for the pure logic in the rag package (no network / no DB)."""
import pytest

from rag.config import Settings
from rag.prompts import build_no_rag_prompt, build_rag_prompt, format_context
from rag.retrieval import RETRIEVAL_METHODS


# -- prompts ----------------------------------------------------------------

def test_format_context_numbers_documents():
    docs = [
        {"question": "Capital of France?", "answer": "Paris"},
        {"question": "FIFA 2018 winner?", "answer": "France"},
    ]
    ctx = format_context(docs)
    assert "[1]" in ctx and "[2]" in ctx
    assert "Paris" in ctx and "France" in ctx


def test_format_context_handles_empty():
    assert "no references" in format_context([]).lower()


def test_build_rag_prompt_includes_query_and_context():
    prompt = build_rag_prompt("Who wrote Hamlet?", [{"question": "x", "answer": "Shakespeare"}])
    assert "Who wrote Hamlet?" in prompt
    assert "Shakespeare" in prompt


def test_no_rag_prompt_has_no_context():
    prompt = build_no_rag_prompt("Who wrote Hamlet?")
    assert "Who wrote Hamlet?" in prompt
    assert "reference" not in prompt.lower()


# -- retrieval registry -----------------------------------------------------

def test_registry_has_all_four_strategies():
    assert set(RETRIEVAL_METHODS) == {
        "semantic",
        "bm25",
        "hybrid_alpha_0.8",
        "semantic_plus_rerank",
    }
    assert all(callable(fn) for fn in RETRIEVAL_METHODS.values())


# -- config -----------------------------------------------------------------

def test_settings_require_raises_on_missing():
    settings = Settings(
        openrouter_api_key=None,
        cohere_api_key=None,
        huggingface_api_key=None,
        phoenix_api_key=None,
        weaviate_url=None,
        weaviate_api_key=None,
        workspace_id=None,
    )
    with pytest.raises(RuntimeError, match="openrouter_api_key"):
        settings.require("openrouter_api_key")


def test_settings_require_passes_when_present():
    settings = Settings(
        openrouter_api_key="key",
        cohere_api_key=None,
        huggingface_api_key=None,
        phoenix_api_key=None,
        weaviate_url=None,
        weaviate_api_key=None,
        workspace_id=None,
    )
    settings.require("openrouter_api_key")  # should not raise
