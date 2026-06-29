"""Multi-strategy Retrieval-Augmented Generation toolkit.

A small, importable package extracted from the project notebook so the core
logic can be read, reused, and tested independently of Jupyter:

    config      - environment-driven settings
    llm         - provider-agnostic (OpenAI-compatible) chat client
    retrieval   - the four retrieval strategies + a name->function registry
    prompts     - prompt construction for RAG and the no-RAG baseline
    pipeline    - end-to-end RAG pipeline with latency instrumentation
    evaluation  - LLM-as-a-judge metrics and a multi-strategy benchmark loop
"""

from .config import Settings, load_settings
from .llm import LLMClient
from .retrieval import (
    RETRIEVAL_METHODS,
    bm25_retrieve,
    hybrid_retrieve,
    semantic_search_retrieve,
    semantic_search_with_reranking,
)
from .prompts import build_rag_prompt, build_no_rag_prompt
from .pipeline import RAGPipeline, RAGResult

__all__ = [
    "Settings",
    "load_settings",
    "LLMClient",
    "RETRIEVAL_METHODS",
    "bm25_retrieve",
    "hybrid_retrieve",
    "semantic_search_retrieve",
    "semantic_search_with_reranking",
    "build_rag_prompt",
    "build_no_rag_prompt",
    "RAGPipeline",
    "RAGResult",
]
