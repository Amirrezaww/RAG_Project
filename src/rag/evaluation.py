"""LLM-as-a-judge evaluation and a multi-strategy benchmark loop.

Two judges are provided:

* ``judge_relevance`` -- is the retrieved context useful for the query?
* ``judge_faithfulness`` -- is the answer supported by the retrieved context?

``benchmark_strategies`` runs every retrieval method in a registry across a set
of queries and returns a tidy DataFrame of latency + quality scores, ready to
group and plot.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .llm import LLMClient
from .pipeline import RAGPipeline
from .retrieval import RETRIEVAL_METHODS, RetrieveFn

_YES = "YES"


def _binary_judge(llm: LLMClient, question: str) -> float:
    """Ask the judge a YES/NO question; return 1.0 for YES else 0.0."""
    verdict = llm.generate(question, max_tokens=5).strip().upper()
    return 1.0 if _YES in verdict else 0.0


def judge_relevance(llm: LLMClient, query: str, context: str) -> float:
    """1.0 if the retrieved context is relevant to the query, else 0.0."""
    prompt = (
        f"Query: {query}\nRetrieved context: {context}\n\n"
        "Is the retrieved context relevant to answering the query? "
        "Respond ONLY with 'YES' or 'NO'."
    )
    return _binary_judge(llm, prompt)


def judge_faithfulness(llm: LLMClient, context: str, answer: str) -> float:
    """1.0 if the answer is supported by the context, else 0.0."""
    prompt = (
        f"Context: {context}\nAnswer: {answer}\n\n"
        "Is the answer supported by the context provided? "
        "Respond ONLY with 'YES' or 'NO'."
    )
    return _binary_judge(llm, prompt)


def benchmark_strategies(
    queries: list[dict[str, Any]],
    pipeline: RAGPipeline,
    judge: LLMClient,
    methods: dict[str, RetrieveFn] | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """Benchmark every retrieval strategy across ``queries``.

    Each query dict needs a ``query`` key (``ground_truth`` is optional). Returns
    one row per (method, query) with latency, relevance, and faithfulness.
    """
    methods = methods or RETRIEVAL_METHODS
    rows: list[dict[str, Any]] = []

    for method_name, retrieve_fn in methods.items():
        for item in queries:
            query = item["query"]
            result = pipeline.run(query, retrieve_fn, top_k=top_k)
            context = result.context_text()
            rows.append(
                {
                    "method": method_name,
                    "query": query,
                    "latency_s": result.total_latency_ms / 1000,
                    "relevance": judge_relevance(judge, query, context),
                    "faithfulness": judge_faithfulness(judge, context, result.response),
                    "response_length": len(result.response),
                }
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Average latency / relevance / faithfulness grouped by method."""
    return df.groupby("method")[["latency_s", "relevance", "faithfulness"]].mean()
