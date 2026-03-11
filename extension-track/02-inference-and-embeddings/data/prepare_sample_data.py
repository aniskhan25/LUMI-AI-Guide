#!/usr/bin/env python3
"""Regenerate sample datasets for Lesson 02."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CORPUS_TEXTS = [
    ("energy", "Hydropower dispatch plan includes weekly balancing and reserve margins."),
    ("healthcare", "Radiology report queue grew after weekend backlog and staffing gaps."),
    ("finance", "Fraud monitoring alerts increased for transactions from new merchant categories."),
    ("retail", "Catalog cleanup removed duplicate SKUs and fixed inconsistent product tags."),
    ("industrial", "Pump station telemetry shows periodic pressure drops during high-demand windows."),
    ("security", "Identity provider logs indicate repeated failed MFA attempts from unknown IP ranges."),
    ("education", "Student support tickets ask for clearer assignment grading criteria."),
    ("transport", "Freight schedule optimization reduced idle time at transfer hubs."),
    ("research", "Experiment notes compare tokenization strategies for multilingual corpora."),
    ("telecom", "Service outage report points to fiber cut and delayed failover trigger."),
]

GEN_PROMPTS = [
    "Summarize in one sentence: Input data validation failed for 4 percent of records.",
    "Rewrite as a task: Confirm that all output IDs match the original input IDs.",
    "Generate a status line: Batch inference job completed without GPU memory errors.",
    "Create a concise update: Throughput improved after increasing batch size from 8 to 16.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Directory for JSONL files")
    parser.add_argument("--corpus-size", type=int, default=20)
    parser.add_argument("--generation-size", type=int, default=8)
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    corpus_rows = []
    for i in range(args.corpus_size):
        domain, text = CORPUS_TEXTS[i % len(CORPUS_TEXTS)]
        corpus_rows.append(
            {
                "id": f"doc-{i+1:04d}",
                "text": text,
                "metadata": {"domain": domain, "lang": "en"},
            }
        )

    generation_rows = []
    for i in range(args.generation_size):
        generation_rows.append({"id": f"gen-{i+1:04d}", "prompt": GEN_PROMPTS[i % len(GEN_PROMPTS)]})

    corpus_path = args.output / "sample_corpus.jsonl"
    generation_path = args.output / "sample_generation_inputs.jsonl"

    write_jsonl(corpus_path, corpus_rows)
    write_jsonl(generation_path, generation_rows)

    print(f"WROTE={corpus_path}")
    print(f"WROTE={generation_path}")


if __name__ == "__main__":
    main()

