"""Regenerate the project's derived datasets from their public HuggingFace sources.

The committed repo does NOT ship the raw corpora (they are large and reproducible).
Running this script downloads Natural Questions (`nq_open`) and HotpotQA
(`hotpot_qa`, distractor) and writes the same artifacts the notebook expects into
`data/`:

    data/nq_documents.jsonl       - unique answer-string documents (NQ)
    data/nq_qa_pairs.pkl          - [{question, answers}, ...]
    data/hotpot_documents.jsonl   - unique context sentences (HotpotQA, ~320 MB)
    data/hotpot_qa_pairs.pkl      - [{question, answer, type}, ...]

Usage:
    python scripts/download_data.py --max-examples 85000

Requires the `datasets` package (see requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _write_jsonl(path: Path, documents: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def build_nq(max_examples: int) -> None:
    print(f"Loading nq_open (up to {max_examples} train examples)...")
    nq = load_dataset("nq_open")
    df = nq["train"].to_pandas()

    documents: list[str] = []
    qa_pairs: list[dict] = []
    for _, row in df.head(max_examples).iterrows():
        answers = row["answer"]
        if len(answers) == 0:
            continue
        for ans in answers:
            if isinstance(ans, str) and ans.strip():
                documents.append(ans.strip())
        qa_pairs.append({"question": row["question"], "answers": list(answers)})

    documents = list(OrderedDict.fromkeys(documents))
    _write_jsonl(DATA_DIR / "nq_documents.jsonl", documents)
    with (DATA_DIR / "nq_qa_pairs.pkl").open("wb") as f:
        pickle.dump(qa_pairs, f)
    print(f"  -> {len(documents)} NQ documents, {len(qa_pairs)} QA pairs")


def build_hotpot(max_examples: int) -> None:
    print(f"Loading hotpot_qa/distractor (up to {max_examples} train examples)...")
    hotpot = load_dataset("hotpot_qa", "distractor")
    n = min(max_examples, len(hotpot["train"]))

    documents: list[str] = []
    qa_pairs: list[dict] = []
    for example in hotpot["train"].select(range(n)):
        context = example.get("context", {})
        if isinstance(context, dict) and "sentences" in context:
            for sentence_list in context["sentences"]:
                if isinstance(sentence_list, list):
                    for s in sentence_list:
                        if isinstance(s, str) and s.strip():
                            documents.append(s.strip())
        qa_pairs.append(
            {
                "question": example.get("question", ""),
                "answer": example.get("answer", ""),
                "type": example.get("type"),
            }
        )

    documents = list(OrderedDict.fromkeys(documents))
    _write_jsonl(DATA_DIR / "hotpot_documents.jsonl", documents)
    with (DATA_DIR / "hotpot_qa_pairs.pkl").open("wb") as f:
        pickle.dump(qa_pairs, f)
    print(f"  -> {len(documents)} HotpotQA documents, {len(qa_pairs)} QA pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-examples", type=int, default=85000,
                        help="Cap on train examples processed per dataset.")
    parser.add_argument("--dataset", choices=["nq", "hotpot", "all"], default="all")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("nq", "all"):
        build_nq(args.max_examples)
    if args.dataset in ("hotpot", "all"):
        build_hotpot(args.max_examples)
    print("Done.")


if __name__ == "__main__":
    main()
