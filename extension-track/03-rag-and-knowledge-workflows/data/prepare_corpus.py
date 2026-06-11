#!/usr/bin/env python3
"""Prepare a small document corpus and query set for Lesson 03."""

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
    {
        "doc_id": "doc-005",
        "title": "Procurement Workflow",
        "text": "Procurement requests above the defined threshold require dual approval from the budget owner and compliance officer. Missing approvals invalidate purchase orders and trigger a manual review.",
        "metadata": {"domain": "finance", "version": "v2"},
    },
    {
        "doc_id": "doc-006",
        "title": "Network Capacity Planning",
        "text": "Traffic baselines are recalculated monthly using peak-hour observations. Capacity plans must include failover overhead and at least twenty percent growth margin for the next planning period.",
        "metadata": {"domain": "network", "version": "v1"},
    },
    {
        "doc_id": "doc-007",
        "title": "Laboratory Safety Protocol",
        "text": "Personnel must wear eye protection and gloves in all wet-lab zones. Chemical containers require readable labels and storage by compatibility group. Spills are reported immediately and cleaned using approved kits.",
        "metadata": {"domain": "safety", "version": "v4"},
    },
    {
        "doc_id": "doc-008",
        "title": "Service-Level Objectives",
        "text": "Core API endpoints target p95 latency below 250 milliseconds and monthly availability above 99.9 percent. Breaches require root-cause analysis and a mitigation plan reviewed in weekly operations meetings.",
        "metadata": {"domain": "platform", "version": "v1"},
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
    {
        "query_id": "q-004",
        "query": "What approvals are needed for high-value procurement requests?",
    },
    {
        "query_id": "q-005",
        "query": "What are the API latency and availability objectives?",
    },
    {
        "query_id": "q-006",
        "query": "What are the required steps when coolant temperature remains too high?",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    corpus_path = args.output / "corpus.jsonl"
    queries_path = args.output / "queries.jsonl"

    write_jsonl(corpus_path, CORPUS)
    write_jsonl(queries_path, QUERIES)

    print(f"WROTE={corpus_path}")
    print(f"WROTE={queries_path}")


if __name__ == "__main__":
    main()
