#!/usr/bin/env python3
"""Prepare a small AG News corpus for embeddings and generation."""

import argparse
import json
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit("datasets is required to prepare AG News.") from exc


LABELS = {
    0: "world",
    1: "sports",
    2: "business",
    3: "science_technology",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-size", type=int, default=512)
    parser.add_argument("--generation-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("ag_news")["train"].shuffle(seed=args.seed)
    corpus_rows = dataset.select(range(args.corpus_size))
    generation_rows = dataset.select(range(args.generation_size))

    corpus = []
    for i, row in enumerate(corpus_rows, start=1):
        corpus.append(
            {
                "id": f"doc-{i:04d}",
                "text": str(row["text"]),
                "metadata": {"label": int(row["label"]), "category": LABELS[int(row["label"])]},
            }
        )

    prompts = []
    for i, row in enumerate(generation_rows, start=1):
        prompts.append(
            {
                "id": f"gen-{i:04d}",
                "prompt": f"Summarize this news item in one sentence:\n{row['text']}",
            }
        )

    corpus_path = args.output / "ag_news_corpus.jsonl"
    prompts_path = args.output / "ag_news_generation_inputs.jsonl"
    write_jsonl(corpus_path, corpus)
    write_jsonl(prompts_path, prompts)

    print(f"WROTE={corpus_path}")
    print(f"WROTE={prompts_path}")


if __name__ == "__main__":
    main()
