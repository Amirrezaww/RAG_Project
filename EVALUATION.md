# Evaluation methodology & findings

This note documents how the RAG system is evaluated, what the headline numbers
mean, and — importantly — the two methodological issues found while building it
and how they were handled. Being explicit about these is the point: a RAG
evaluation is only as trustworthy as its judge and its prompts.

## How the system is scored

Two complementary evaluation paths are used:

1. **RAG vs. No-RAG (correctness).** The same generator (Llama-3.2-3B-Instruct)
   answers a 100-query sample from Natural Questions twice: once with retrieved
   context, once from parametric knowledge alone. A stronger model
   (`gpt-5-mini`) acts as a QA judge scoring answer **accuracy** against the
   ground-truth answer, and **faithfulness** (is the answer grounded in the
   retrieved context?).

2. **Per-strategy benchmark.** Each of the four retrieval strategies answers a
   shared 20-query sample; an LLM judge scores **relevance** (is the retrieved
   context useful?) and **faithfulness**, alongside measured latency.

> **The two paths do not use the same judge.** Path 1 is graded by
> `gpt-5-mini` — a far stronger model than the generator, which is the correct
> setup. Path 2 uses `llama-3.2-3b-instruct`, i.e. the generator grading its own
> output. See Issue 3 below.

| Metric        | Question it answers                                  | Reference used      |
|---------------|------------------------------------------------------|---------------------|
| Accuracy      | Is the final answer correct?                         | ground-truth answer |
| Relevance     | Is the retrieved context useful for the query?       | retrieved context   |
| Faithfulness  | Is the answer supported by the retrieved context?    | retrieved context   |

## Headline result

On the 100-query NQ sample, retrieval more than doubled accuracy:

| System          | Accuracy | Faithfulness | Latency |
|-----------------|:--------:|:------------:|:-------:|
| RAG             | 77.8%    | 61.0%        | 8.55 s  |
| No-RAG baseline | 32.0%    | —            | 6.47 s  |

This comparison is robust: both systems share the generator, the judge, and the
query set, so the 45-point accuracy gap is attributable to retrieval.

## Issue 1 — prompt leakage in generated answers (fixed)

**Symptom.** Many RAG answers opened by narrating the prompt, e.g.
*"Based on the additional information provided, I will update my knowledge to
reflect that…"* before giving the actual answer. This inflates verbosity and can
confuse the judge.

**Root cause.** The original generation prompt told the model the context was
*"from 2024 … add it to your overall knowledge"* and framed it as *"2024 News"*.
A small 3B model takes that framing literally and narrates the act of
incorporating the information.

**Fix.** The prompt was rewritten to be directive and source-neutral: use the
references when relevant, fall back on parametric knowledge otherwise, and
**"reply with a direct, concise answer only — do not narrate your reasoning or
mention that information was provided."** This lives in
[`src/rag/prompts.py`](src/rag/prompts.py) (`build_rag_prompt`) and the notebook's
`generate_final_prompt`. Results in the committed CSV predate the fix and should
be regenerated (`python scripts/download_data.py` then re-run the eval cells)
to quantify the improvement.

## Issue 2 — faithfulness is ill-posed on the NQ corpus (documented limitation)

**Symptom.** Faithfulness scores are near zero for most strategies, even when
answers are correct.

**Root cause.** The Natural Questions corpus, as indexed here, stores **answers**
(often a single entity like *"France"*) rather than supporting **passages**.
Asking *"is the answer grounded in the retrieved text?"* against a corpus of bare
answer strings is not well defined — there is no passage to ground against. The
low faithfulness numbers reflect the corpus design, not hallucination.

**Path forward.** Index a passage-level corpus (e.g. Wikipedia paragraphs, which
HotpotQA already provides via its `context.sentences`) so faithfulness measures
real grounding. The HotpotQA ingestion path in the notebook already extracts
these sentences; pointing the faithfulness judge at them is the natural next
experiment.

## Issue 3 — the per-strategy comparison uses a self-judge (open)

**Symptom.** The four-strategy table shows low relevance across the board and
near-zero faithfulness, with hybrid and reranking scoring *below* plain BM25 —
the opposite of what the retrieval literature would predict.

**Root cause.** `calculate_metrics()` grades with
`meta-llama/llama-3.2-3b-instruct` — the same 3B model that produced the
answers. A model that small is unreliable at binary YES/NO grading, and it is
being asked to judge its own output. By contrast, the headline RAG vs. No-RAG
result is graded by `gpt-5-mini`, which is the methodologically sound setup.

**Consequence.** The *relative ranking* of the four strategies is not
trustworthy. The headline result is unaffected — it uses the strong judge, and
both arms of that comparison share the same judge, generator, and query set.

**Path forward.** Re-run the four-strategy benchmark with `gpt-5-mini` as the
judge (the same evaluator objects used in path 1) and a larger sample. This is
the single highest-value next experiment: it is cheap, and it would either
confirm or overturn the current strategy ranking. `benchmark_strategies()` in
[`src/rag/evaluation.py`](src/rag/evaluation.py) already accepts the judge as a
separate `LLMClient` argument, so a stronger judge can be swapped in directly.

## Reproducing the evaluation

```bash
pip install -r requirements.txt
cp .env.example API.env          # add your keys
python scripts/download_data.py  # rebuild datasets from HuggingFace
jupyter lab RAG.ipynb            # run sections 8–10 for tracing + evaluation
```

The `src/rag` package mirrors this logic for programmatic use; see
`benchmark_strategies` in [`src/rag/evaluation.py`](src/rag/evaluation.py).
