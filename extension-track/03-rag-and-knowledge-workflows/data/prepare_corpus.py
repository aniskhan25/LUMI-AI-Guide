#!/usr/bin/env python3
"""Generate sample corpus and query files for Lesson 03."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CORPUS = [
    {
        "doc_id": "doc-001",
        "title": "Cooling System Maintenance",
        "text": "The cooling system requires weekly inspection of pump seals, coolant levels, and fan operation. If coolant temperature exceeds threshold values for more than ten minutes, operators must trigger the emergency cooling protocol and log the incident with timestamp and unit identifier.",
        "metadata": {"domain": "operations", "version": "v1"},
    },
    {
        "doc_id": "doc-002",
        "title": "Incident Response Escalation",
        "text": "Security incidents are classified into low, medium, and high severity. High severity incidents require immediate escalation to the on-call lead, containment actions within fifteen minutes, and a post-incident report within twenty-four hours.",
        "metadata": {"domain": "security", "version": "v2"},
    },
    {
        "doc_id": "doc-003",
        "title": "Data Retention Policy",
        "text": "Customer support transcripts are retained for twelve months in encrypted storage. Access is restricted to approved teams. Deletion requests must be processed within thirty days and documented in the audit log.",
        "metadata": {"domain": "governance", "version": "v1"},
    },
    {
        "doc_id": "doc-004",
        "title": "Model Validation Checklist",
        "text": "Before deployment, each model release must pass schema checks, latency checks, and regression tests on a fixed benchmark set. Any release failing a critical check is blocked until remediation and re-validation are complete.",
        "metadata": {"domain": "mlops", "version": "v3"},
    },
]

QUERIES = [
    {
        "query_id": "q-001",
        "query": "What actions are required when a high severity security incident occurs?",
    },
    {
        "query_id": "q-002",
        "query": "How long are customer support transcripts retained and how are deletions handled?",
    },
    {
        "query_id": "q-003",
        "query": "What checks must a model pass before deployment?",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Output directory for sample JSONL files")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    corpus_path = args.output / "sample_corpus.jsonl"
    queries_path = args.output / "sample_queries.jsonl"

    write_jsonl(corpus_path, CORPUS)
    write_jsonl(queries_path, QUERIES)

    print(f"WROTE={corpus_path}")
    print(f"WROTE={queries_path}")


if __name__ == "__main__":
    main()

