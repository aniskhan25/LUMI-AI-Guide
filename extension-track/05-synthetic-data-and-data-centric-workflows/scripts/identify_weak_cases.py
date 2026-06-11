#!/usr/bin/env python3
"""Select weak cases from Lesson 04 failures or the checked-in fallback set."""

import argparse
import json
from pathlib import Path

from _common import load_yaml, read_jsonl, resolve_path, resolve_run_dir, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def from_failure_samples(rows):
    weak_cases = []
    for row in rows:
        query_id = str(row.get("query_id", ""))
        question = str(row.get("question", "")).strip()
        if not query_id or not question:
            continue
        reference_answer = str(row.get("reference_answer", "")).strip()
        required_terms = [term for term in reference_answer.replace(",", " ").split() if len(term) > 4][:5]
        weak_cases.append(
            {
                "case_id": f"wc-from-{query_id}",
                "input_text": question,
                "failure_type": str(row.get("failure_category", "unknown")),
                "gap_label": str(row.get("failure_category", "unknown")),
                "evidence_reference": str(row.get("expected_doc_id", "")),
                "reference_answer": reference_answer,
                "required_terms": required_terms,
            }
        )
    return weak_cases


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)

    weak_cases_path = resolve_path(args.config.parent, str(cfg["paths"]["weak_cases_jsonl"]))
    failure_path = resolve_path(args.config.parent, str(cfg["paths"]["lesson4_failure_samples_jsonl"]))
    selected_path = run_dir / str(cfg["output"]["selected_weak_cases_jsonl"])

    source = "weak_cases"
    if failure_path.is_file():
        failure_rows = read_jsonl(failure_path)
        weak_cases = from_failure_samples(failure_rows)
        if weak_cases:
            source = "lesson4_failure_samples"
        else:
            weak_cases = read_jsonl(weak_cases_path)
    else:
        weak_cases = read_jsonl(weak_cases_path)

    if args.max_cases > 0:
        weak_cases = weak_cases[: args.max_cases]
    if not weak_cases:
        raise SystemExit("No weak cases available from configured sources.")

    write_jsonl(selected_path, weak_cases)
    summary = {
        "source": source,
        "selected_count": len(weak_cases),
        "selected_path": str(selected_path),
    }
    summary_path = run_dir / str(cfg["output"]["weak_case_selection_summary_json"])
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"WEAK_CASE_SOURCE={source}")
    print(f"SELECTED_COUNT={len(weak_cases)}")
    print(f"SELECTED_PATH={selected_path}")


if __name__ == "__main__":
    main()
