#!/usr/bin/env python3
"""Select weak cases from configured source or fallback dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _common import load_yaml, read_jsonl, resolve_path, resolve_run_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all")
    return parser.parse_args()


def from_failure_samples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    weak: List[Dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("query_id", ""))
        question = str(row.get("question", "")).strip()
        if not qid or not question:
            continue
        ref = str(row.get("reference_answer", "")).strip()
        terms = [t for t in ref.replace(",", " ").split() if len(t) > 4][:5]
        weak.append(
            {
                "case_id": f"wc-from-{qid}",
                "input_text": question,
                "failure_type": str(row.get("failure_category", "unknown")),
                "gap_label": str(row.get("failure_category", "unknown")),
                "evidence_reference": str(row.get("expected_doc_id", "")),
                "reference_answer": ref,
                "required_terms": terms,
            }
        )
    return weak


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.generate_config)
    run_dir = resolve_run_dir(cfg, args.generate_config, args.output_root, args.run_name)

    weak_cases_path = resolve_path(args.generate_config.parent, str(cfg["paths"]["weak_cases_jsonl"]))
    failure_path = resolve_path(args.generate_config.parent, str(cfg["paths"]["lesson4_failure_samples_jsonl"]))
    selected_path = run_dir / str(cfg["output"]["selected_weak_cases_jsonl"])

    source = "weak_cases"
    weak_cases: List[Dict[str, Any]]
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
    summary_path = run_dir / "weak_case_selection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"WEAK_CASE_SOURCE={source}")
    print(f"SELECTED_COUNT={len(weak_cases)}")
    print(f"SELECTED_PATH={selected_path}")


if __name__ == "__main__":
    main()

