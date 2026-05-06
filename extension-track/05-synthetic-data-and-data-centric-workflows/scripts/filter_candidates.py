#!/usr/bin/env python3
"""Filter and curate generated synthetic candidates."""

import argparse
import json
from pathlib import Path

from _common import load_yaml, qa_key, read_jsonl, resolve_path, resolve_run_dir, term_overlap_score, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)

    baseline_path = resolve_path(args.config.parent, str(cfg["paths"]["baseline_dataset_jsonl"]))
    candidates_path = run_dir / str(cfg["output"]["candidates_jsonl"])
    baseline_rows = read_jsonl(baseline_path)
    candidate_rows = read_jsonl(candidates_path)
    if not candidate_rows:
        raise SystemExit(f"No candidates found at {candidates_path}. Run generate_candidates.py first.")

    rules = cfg["filter"]
    min_input_chars = int(rules["min_input_chars"])
    min_target_chars = int(rules["min_target_chars"])
    require_terms = bool(rules["require_required_term_match"])
    dedup_baseline = bool(rules["deduplicate_against_baseline"])
    dedup_candidates = bool(rules["deduplicate_candidates"])

    baseline_keys = {qa_key(str(row.get("question", "")), str(row.get("answer", ""))) for row in baseline_rows}

    accepted = []
    rejected = []
    filtered = []
    seen_accepted_keys = set()

    for row in candidate_rows:
        reasons = []
        question = str(row.get("generated_input", "")).strip()
        answer = str(row.get("generated_target", "")).strip()
        required_terms = [str(x) for x in row.get("required_terms", [])]
        key = qa_key(question, answer)

        if not row.get("synthetic_id"):
            reasons.append("missing_synthetic_id")
        if len(question) < min_input_chars:
            reasons.append("input_too_short")
        if len(answer) < min_target_chars:
            reasons.append("target_too_short")
        if require_terms and required_terms and term_overlap_score(answer, required_terms) <= 0.0:
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

    filter_all_path = run_dir / str(cfg["output"]["filter_all_jsonl"])
    accepted_path = run_dir / str(cfg["output"]["accepted_jsonl"])
    rejected_path = run_dir / str(cfg["output"]["rejected_jsonl"])
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
    summary_path = run_dir / str(cfg["output"]["filter_summary_json"])
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"CANDIDATES={len(candidate_rows)}")
    print(f"ACCEPTED={len(accepted)}")
    print(f"REJECTED={len(rejected)}")
    print(f"SUMMARY_PATH={summary_path}")


if __name__ == "__main__":
    main()
