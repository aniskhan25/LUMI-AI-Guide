#!/usr/bin/env python3
"""Filter and curate generated synthetic candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _common import load_yaml, qa_key, read_jsonl, resolve_path, resolve_run_dir, term_overlap_score, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-config", type=Path, required=True)
    parser.add_argument("--filter-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gcfg = load_yaml(args.generate_config)
    fcfg = load_yaml(args.filter_config)
    run_dir = resolve_run_dir(gcfg, args.generate_config, args.output_root, args.run_name)

    baseline_path = resolve_path(args.generate_config.parent, str(gcfg["paths"]["baseline_dataset_jsonl"]))
    candidates_path = run_dir / str(gcfg["output"]["candidates_jsonl"])
    baseline_rows = read_jsonl(baseline_path)
    candidate_rows = read_jsonl(candidates_path)

    if not candidate_rows:
        raise SystemExit(f"No candidates found at {candidates_path}. Run generate_candidates.py first.")

    rules = fcfg["rules"]
    min_input_chars = int(rules["min_input_chars"])
    min_target_chars = int(rules["min_target_chars"])
    require_terms = bool(rules["require_required_term_match"])
    dedup_baseline = bool(rules["deduplicate_against_baseline"])
    dedup_candidates = bool(rules["deduplicate_candidates"])

    baseline_keys = set()
    for row in baseline_rows:
        baseline_keys.add(qa_key(str(row.get("question", "")), str(row.get("answer", ""))))

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []
    seen_accepted_keys = set()

    for row in candidate_rows:
        reasons: List[str] = []
        q = str(row.get("generated_input", "")).strip()
        a = str(row.get("generated_target", "")).strip()
        terms = [str(x) for x in row.get("required_terms", [])]
        key = qa_key(q, a)

        if not row.get("synthetic_id"):
            reasons.append("missing_synthetic_id")
        if len(q) < min_input_chars:
            reasons.append("input_too_short")
        if len(a) < min_target_chars:
            reasons.append("target_too_short")
        if require_terms and terms and term_overlap_score(a, terms) <= 0.0:
            reasons.append("no_required_term_match")
        if dedup_baseline and key in baseline_keys:
            reasons.append("duplicate_of_baseline")
        if dedup_candidates and key in seen_accepted_keys:
            reasons.append("duplicate_in_candidates")

        out = dict(row)
        if reasons:
            out["filter_status"] = "rejected"
            out["filter_reasons"] = reasons
            rejected.append(out)
        else:
            out["filter_status"] = "accepted"
            out["filter_reasons"] = []
            accepted.append(out)
            seen_accepted_keys.add(key)
        filtered.append(out)

    filter_all_path = run_dir / str(gcfg["output"]["filter_all_jsonl"])
    accepted_path = run_dir / str(gcfg["output"]["accepted_jsonl"])
    rejected_path = run_dir / str(gcfg["output"]["rejected_jsonl"])
    write_jsonl(filter_all_path, filtered)
    write_jsonl(accepted_path, accepted)
    write_jsonl(rejected_path, rejected)

    summary = {
        "candidate_count": len(candidate_rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "acceptance_rate": len(accepted) / max(1, len(candidate_rows)),
        "filter_all_path": str(filter_all_path),
        "accepted_path": str(accepted_path),
        "rejected_path": str(rejected_path),
    }
    summary_path = run_dir / str(gcfg["output"]["filter_summary_json"])
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"CANDIDATES={len(candidate_rows)}")
    print(f"ACCEPTED={len(accepted)}")
    print(f"REJECTED={len(rejected)}")
    print(f"SUMMARY_PATH={summary_path}")


if __name__ == "__main__":
    main()

