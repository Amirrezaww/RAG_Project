"""End-to-end RAG pipeline with latency instrumentation.

``RAGPipeline`` wires a retrieval strategy to the LLM client and returns a
structured ``RAGResult`` carrying the answer, the retrieved context, and a
retrieval/generation latency breakdown. The same object also exposes a no-RAG
baseline so the two can be compared on identical inputs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient
from .prompts import build_no_rag_prompt, build_rag_prompt
from .retrieval import RetrieveFn


@dataclass
class RAGResult:
    """Outcome of a single pipeline run."""

    query: str
    response: str
    contexts: list[str]
    retrieved_questions: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    num_contexts: int = 0

    def context_text(self) -> str:
        """Retrieved answers joined into a single reference string."""
        return "\n".join(self.contexts)


class RAGPipeline:
    """Retrieve-then-generate, with timing on each stage."""

    def __init__(self, llm: LLMClient, collection: Any) -> None:
        self.llm = llm
        self.collection = collection

    def run(
        self,
        query: str,
        retrieve_function: RetrieveFn,
        top_k: int = 5,
    ) -> RAGResult:
        """Run retrieval + generation for ``query`` and time both stages."""
        start = time.perf_counter()

        retrieve_start = time.perf_counter()
        documents = retrieve_function(query, self.collection, top_k=top_k)
        retrieval_latency = (time.perf_counter() - retrieve_start) * 1000

        prompt = build_rag_prompt(query, documents)

        gen_start = time.perf_counter()
        response = self.llm.generate(prompt)
        generation_latency = (time.perf_counter() - gen_start) * 1000

        total_latency = (time.perf_counter() - start) * 1000
        return RAGResult(
            query=query,
            response=response,
            contexts=[doc.get("answer", "") for doc in documents],
            retrieved_questions=[doc.get("question", "") for doc in documents],
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
            total_latency_ms=total_latency,
            num_contexts=len(documents),
        )

    def run_no_rag(self, query: str) -> RAGResult:
        """Baseline: answer ``query`` from the model's parametric knowledge only."""
        start = time.perf_counter()
        response = self.llm.generate(build_no_rag_prompt(query))
        total_latency = (time.perf_counter() - start) * 1000
        return RAGResult(
            query=query,
            response=response,
            contexts=[],
            generation_latency_ms=total_latency,
            total_latency_ms=total_latency,
            num_contexts=0,
        )
