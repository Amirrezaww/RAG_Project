# Multi-Strategy RAG Pipeline

A production-oriented Retrieval-Augmented Generation system that evaluates four retrieval strategies against a no-RAG baseline — built to answer knowledge-intensive questions from large document corpora with measurable accuracy.

Built as an MSc dissertation project (awarded Top 5 university-wide), this goes beyond a standard RAG tutorial: it implements and systematically benchmarks competing retrieval architectures, integrates real-time LLM monitoring, and ships an interactive comparison UI.

---

## What it does

Given a user query, the system retrieves relevant documents using one of four strategies, then passes them to Llama 3.2 to generate a grounded, cited answer. All five approaches (including no-RAG baseline) run simultaneously and their outputs are displayed side by side for direct comparison.

User Query

│

▼

┌─────────────────────────────────────────────────┐

│              Weaviate Vector Database            │

│                                                 │

│  ┌──────────┐ ┌──────┐ ┌────────┐ ┌─────────┐  │

│  │ Semantic │ │ BM25 │ │Hybrid  │ │Semantic │  │

│  │  (Dense) │ │      │ │        │ │+ Rerank │  │

│  └──────────┘ └──────┘ └────────┘ └─────────┘  │

└─────────────────────────────────────────────────┘

│

▼

Llama 3.2 (via OpenRouter / Together AI)

│

▼

Grounded Answer  +  Arize Phoenix Monitoring

---

## Retrieval Strategies

| Strategy | Description |
|---|---|
| **Semantic Search** | Dense vector similarity using `BAAI/bge-base-en-v1.5` embeddings |
| **BM25** | Keyword-based sparse retrieval — strong on exact matches and rare terms |
| **Hybrid** | Weighted combination of semantic and BM25 signals |
| **Semantic + Reranking** | Semantic retrieval followed by Cohere reranking for precision |
| **No RAG (baseline)** | Direct Llama 3.2 generation with no retrieval — measures hallucination baseline |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Vector DB | Weaviate |
| Embeddings | `BAAI/bge-base-en-v1.5` via Together AI |
| Reranking | Cohere Rerank |
| LLM | Llama 3.2 (3B Instruct) via OpenRouter / Together AI |
| Monitoring | Arize Phoenix (real-time trace visibility) |
| Interface | Interactive Jupyter widget (ipywidgets) |
| Backend utility | Flask, Python |

---

## Interactive Comparison UI

The notebook ships with a live widget that lets you type any query, set Top-K, choose a reranking property, and see all five strategy outputs rendered side-by-side in real time — no manual switching between cells.

![Widget layout: Semantic | Semantic+Rerank | BM25 on top row, Hybrid | No-RAG on second row]

---

## Monitoring

All LLM traces are logged to Arize Phoenix for real-time observability:
- Latency per retrieval strategy
- Token usage
- Faithfulness and relevance scoring
- Query-level trace inspection

Live dashboard: [app.phoenix.arize.com/s/aamini8118](https://app.phoenix.arize.com/s/aamini8118)

---

## Quickstart

```bash
git clone https://github.com/Amirrezaww/RAG_Project.git
cd RAG_Project
pip install -r requirements.txt
```

Set your API keys:
```bash
export OPENROUTER_API_KEY=your_key_here
export TOGETHER_API_KEY=your_key_here      # optional alternative
```

Then open and run `RAG.ipynb` in Jupyter. The interactive widget loads at the bottom of the notebook.

---

## Key Findings

- Hybrid retrieval consistently outperformed pure semantic and BM25 on knowledge-intensive queries
- Cohere reranking improved precision on ambiguous queries but added ~300ms latency
- No-RAG baseline hallucinated on 60%+ of factual questions that retrieval-augmented strategies answered correctly
- BM25 outperformed semantic search on queries containing rare proper nouns and specific dates

---

## Project Structure---

## Retrieval Strategies

| Strategy | Description |
|---|---|
| **Semantic Search** | Dense vector similarity using `BAAI/bge-base-en-v1.5` embeddings |
| **BM25** | Keyword-based sparse retrieval — strong on exact matches and rare terms |
| **Hybrid** | Weighted combination of semantic and BM25 signals |
| **Semantic + Reranking** | Semantic retrieval followed by Cohere reranking for precision |
| **No RAG (baseline)** | Direct Llama 3.2 generation with no retrieval — measures hallucination baseline |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Vector DB | Weaviate |
| Embeddings | `BAAI/bge-base-en-v1.5` via Together AI |
| Reranking | Cohere Rerank |
| LLM | Llama 3.2 (3B Instruct) via OpenRouter / Together AI |
| Monitoring | Arize Phoenix (real-time trace visibility) |
| Interface | Interactive Jupyter widget (ipywidgets) |
| Backend utility | Flask, Python |

---

## Interactive Comparison UI

The notebook ships with a live widget that lets you type any query, set Top-K, choose a reranking property, and see all five strategy outputs rendered side-by-side in real time — no manual switching between cells.

![Widget layout: Semantic | Semantic+Rerank | BM25 on top row, Hybrid | No-RAG on second row]

---

## Monitoring

All LLM traces are logged to Arize Phoenix for real-time observability:
- Latency per retrieval strategy
- Token usage
- Faithfulness and relevance scoring
- Query-level trace inspection

Live dashboard: [app.phoenix.arize.com/s/aamini8118](https://app.phoenix.arize.com/s/aamini8118)

---

## Quickstart

```bash
git clone https://github.com/Amirrezaww/RAG_Project.git
cd RAG_Project
pip install -r requirements.txt
```

Set your API keys:
```bash
export OPENROUTER_API_KEY=your_key_here
export TOGETHER_API_KEY=your_key_here      # optional alternative
```

Then open and run `RAG.ipynb` in Jupyter. The interactive widget loads at the bottom of the notebook.

---

## Key Findings

- Hybrid retrieval consistently outperformed pure semantic and BM25 on knowledge-intensive queries
- Cohere reranking improved precision on ambiguous queries but added ~300ms latency
- No-RAG baseline hallucinated on 60%+ of factual questions that retrieval-augmented strategies answered correctly
- BM25 outperformed semantic search on queries containing rare proper nouns and specific dates

---

## Project Structure

RAG_Project/

├── RAG.ipynb          # Main notebook — pipeline, experiments, widget

├── utils.py           # LLM clients, embedding functions, interactive widget

├── utils2.py          # Alternative Together AI client, Flask server utilities

├── data/              # Document corpus used for indexing

└── README.md

---

## About

MSc Data Science dissertation — Middlesex University, 2026.
Recognised as Top 5 individual project university-wide.

Built by [Amirreza Amini](https://github.com/Amirrezaww) · [LinkedIn](https://linkedin.com/in/amirreza-amini8118)
