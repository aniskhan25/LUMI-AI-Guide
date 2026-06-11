#!/usr/bin/env python3
"""Rerun the downstream weak-case scoring for baseline and augmented datasets."""

import argparse
import json
from pathlib import Path

from _common import load_yaml, read_jsonl, resolve_path, resolve_run_dir, term_overlap_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def evaluate_dataset(dataset, weak_cases, pass_threshold):
    per_case = []
    total_score = 0.0
    coverage = 0

    for case in weak_cases:
        case_id = str(case["case_id"])
        gap_label = str(case.get("gap_label", "general"))
        required_terms = [str(x) for x in case.get("required_terms", [])]
        rows = [row for row in dataset if str(row.get("gap_label", "")) == gap_label]

        best_score = 0.0
        best_record_id = ""
        for row in rows:
            score = term_overlap_score(str(row.get("answer", "")), required_terms)
            if score > best_score:
                best_score = score
                best_record_id = str(row.get("record_id", ""))

        covered = int(best_score >= pass_threshold)
        coverage += covered
        total_score += best_score
        per_case.append(
            {
                "case_id": case_id,
                "gap_label": gap_label,
                "best_score": best_score,
                "covered": covered,
                "best_record_id": best_record_id,
            }
        )

    case_count = len(weak_cases)
    return {
        "case_count": case_count,
        "avg_case_score": total_score / max(1, case_count),
        "coverage_rate": coverage / max(1, case_count),
        "per_case": per_case,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)

    weak_selected_path = run_dir / str(cfg["output"]["selected_weak_cases_jsonl"])
    if weak_selected_path.is_file():
        weak_cases = read_jsonl(weak_selected_path)
    else:
        weak_cases = read_jsonl(resolve_path(args.config.parent, str(cfg["paths"]["weak_cases_jsonl"])))

    baseline_path = resolve_path(args.config.parent, str(cfg["paths"]["baseline_dataset_jsonl"]))
    augmented_path = run_dir / str(cfg["output"]["augmented_dataset_jsonl"])
    baseline_rows = read_jsonl(baseline_path)
    augmented_rows = read_jsonl(augmented_path)

    pass_threshold = float(cfg["comparison"]["case_pass_threshold"])
    baseline_eval = evaluate_dataset(baseline_rows, weak_cases, pass_threshold)
    baseline_eval["variant"] = "baseline_dataset"
    baseline_eval["dataset_path"] = str(baseline_path)
    baseline_eval["pass_threshold"] = pass_threshold

    augmented_eval = evaluate_dataset(augmented_rows, weak_cases, pass_threshold)
    augmented_eval["variant"] = "augmented_dataset"
    augmented_eval["dataset_path"] = str(augmented_path)
    augmented_eval["pass_threshold"] = pass_threshold

    baseline_out = run_dir / str(cfg["output"]["baseline_rerun_json"])
    augmented_out = run_dir / str(cfg["output"]["augmented_rerun_json"])
    write_json(baseline_out, baseline_eval)
    write_json(augmented_out, augmented_eval)

    print(f"BASELINE_AVG_SCORE={baseline_eval['avg_case_score']:.4f}")
    print(f"AUGMENTED_AVG_SCORE={augmented_eval['avg_case_score']:.4f}")
    print(f"BASELINE_PATH={baseline_out}")
    print(f"AUGMENTED_PATH={augmented_out}")


if __name__ == "__main__":
    main()
