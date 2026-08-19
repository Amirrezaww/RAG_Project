"""Prompt construction for the RAG and no-RAG paths.

The RAG prompt instructs the model to ground its answer in the retrieved
question/answer pairs and to fall back on parametric knowledge only when the
context is insufficient. Keeping prompt text in one place makes it easy to
iterate on (see Phase 3 / evaluation work).
"""
from __future__ import annotations

from typing import Any

Document = dict[str, Any]

_RAG_TEMPLATE = """You are a question-answering assistant. Use the retrieved \
reference passages below to answer the user's question. Prefer the references \
when they are relevant; if they do not contain the answer, say so and answer \
from your own knowledge. Be concise and do not invent sources.

Retrieved references (ordered by relevance):
{context}

Question: {query}
Answer:"""


def format_context(documents: list[Document]) -> str:
    """Render retrieved documents into a compact, numbered context block."""
    lines = []
    for i, doc in enumerate(documents, start=1):
        question = doc.get("question", "")
        answer = doc.get("answer", "")
        lines.append(f"[{i}] Q: {question} | A: {answer}")
    return "\n".join(lines) if lines else "(no references retrieved)"


def build_rag_prompt(query: str, documents: list[Document]) -> str:
    """Build the grounded RAG prompt from a query and retrieved documents."""
    return _RAG_TEMPLATE.format(context=format_context(documents), query=query)


def build_no_rag_prompt(query: str) -> str:
    """Build the no-retrieval baseline prompt (parametric knowledge only)."""
    return f"Answer the following question from your own knowledge: {query}"
