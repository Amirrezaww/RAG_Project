"""The four retrieval strategies, plus a name -> function registry.

Every strategy shares one signature -- ``(query, collection, top_k) -> list[dict]``
-- so the pipeline and benchmark loop can treat them interchangeably. Each
returns the matched objects' property dicts (``question``, ``answer``, ...).
"""
from __future__ import annotations

from typing import Any, Callable

# A retrieved object is its Weaviate property dict.
Document = dict[str, Any]
RetrieveFn = Callable[..., list[Document]]


def filter_by_metadata(
    metadata_property: str, values: list[str], collection: Any, limit: int = 5
) -> list[Document]:
    """Fetch objects whose ``metadata_property`` contains any of ``values``."""
    from weaviate.classes.query import Filter

    response = collection.query.fetch_objects(
        limit=limit,
        filters=Filter.by_property(metadata_property).contains_any(values),
    )
    return [obj.properties for obj in response.objects]


def semantic_search_retrieve(query: str, collection: Any, top_k: int = 5) -> list[Document]:
    """Dense vector search (``near_text``) over the collection's named vector."""
    response = collection.query.near_text(query=query, limit=top_k)
    return [obj.properties for obj in response.objects]


def bm25_retrieve(query: str, collection: Any, top_k: int = 5) -> list[Document]:
    """Sparse keyword search (BM25)."""
    response = collection.query.bm25(query=query, limit=top_k)
    return [obj.properties for obj in response.objects]


def hybrid_retrieve(
    query: str, collection: Any, top_k: int = 5, alpha: float = 0.5
) -> list[Document]:
    """Hybrid fusion of dense and sparse search.

    ``alpha`` weights the two: 0.0 is pure BM25, 1.0 is pure vector search.
    """
    response = collection.query.hybrid(query=query, limit=top_k, alpha=alpha)
    return [obj.properties for obj in response.objects]


def semantic_search_with_reranking(
    query: str,
    collection: Any,
    top_k: int = 5,
    rerank_property: str = "question",
    rerank_query: str | None = None,
) -> list[Document]:
    """Dense search followed by a Cohere cross-encoder rerank on ``rerank_property``."""
    from weaviate.classes.query import Rerank

    reranker = Rerank(query=rerank_query or query, prop=rerank_property)
    response = collection.query.near_text(query=query, limit=top_k, rerank=reranker)
    return [obj.properties for obj in response.objects]


# -- Uniform-signature wrappers so every method is interchangeable -----------

def _hybrid_alpha_08(query: str, collection: Any, top_k: int = 5) -> list[Document]:
    return hybrid_retrieve(query, collection, top_k=top_k, alpha=0.8)


def _semantic_plus_rerank(query: str, collection: Any, top_k: int = 5) -> list[Document]:
    return semantic_search_with_reranking(query, collection, top_k=top_k, rerank_property="question")


#: Registry used by the benchmark loop. Keys are stable, human-readable names.
RETRIEVAL_METHODS: dict[str, RetrieveFn] = {
    "semantic": semantic_search_retrieve,
    "bm25": bm25_retrieve,
    "hybrid_alpha_0.8": _hybrid_alpha_08,
    "semantic_plus_rerank": _semantic_plus_rerank,
}
