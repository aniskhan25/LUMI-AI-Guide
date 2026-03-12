#!/usr/bin/env python3
"""Rerun downstream scoring for baseline and augmented datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _common import load_yaml, read_jsonl, resolve_path, resolve_run_dir, term_overlap_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-config", type=Path, required=True)
    parser.add_argument("--compare-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def evaluate_dataset(dataset: List[Dict[str, Any]], weak_cases: List[Dict[str, Any]], pass_threshold: float) -> Dict[str, Any]:
    per_case: List[Dict[str, Any]] = []
    total_score = 0.0
    coverage = 0

    for case in weak_cases:
        case_id = str(case["case_id"])
        gap = str(case.get("gap_label", "general"))
        required_terms = [str(x) for x in case.get("required_terms", [])]
        rows = [r for r in dataset if str(r.get("gap_label", "")) == gap]

        best_score = 0.0
        best_record_id = ""
        for row in rows:
            score = term_overlap_score(str(row.get("answer", "")), required_terms)
            if score > best_score:
                best_score = score
                best_record_id = str(row.get("record_id", ""))

        is_covered = int(best_score >= pass_threshold)
        coverage += is_covered
        total_score += best_score
        per_case.append(
            {
                "case_id": case_id,
                "gap_label": gap,
                "best_score": best_score,
                "covered": is_covered,
                "best_record_id": best_record_id,
            }
        )

    n = len(weak_cases)
    return {
        "case_count": n,
        "avg_case_score": total_score / max(1, n),
        "coverage_rate": coverage / max(1, n),
        "per_case": per_case,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    gcfg = load_yaml(args.generate_config)
    ccfg = load_yaml(args.compare_config)
    run_dir = resolve_run_dir(gcfg, args.generate_config, args.output_root, args.run_name)

    weak_selected_path = run_dir / str(gcfg["output"]["selected_weak_cases_jsonl"])
    if weak_selected_path.is_file():
        weak_cases = read_jsonl(weak_selected_path)
    else:
        weak_cases_path = resolve_path(args.generate_config.parent, str(gcfg["paths"]["weak_cases_jsonl"]))
        weak_cases = read_jsonl(weak_cases_path)

    baseline_path = resolve_path(args.generate_config.parent, str(gcfg["paths"]["baseline_dataset_jsonl"]))
    augmented_path = run_dir / str(gcfg["output"]["augmented_dataset_jsonl"])
    baseline_rows = read_jsonl(baseline_path)
    augmented_rows = read_jsonl(augmented_path)

    pass_threshold = float(ccfg["evaluation"]["case_pass_threshold"])

    baseline_eval = evaluate_dataset(baseline_rows, weak_cases, pass_threshold)
    baseline_eval["variant"] = "baseline_dataset"
    baseline_eval["dataset_path"] = str(baseline_path)
    baseline_eval["pass_threshold"] = pass_threshold

    augmented_eval = evaluate_dataset(augmented_rows, weak_cases, pass_threshold)
    augmented_eval["variant"] = "augmented_dataset"
    augmented_eval["dataset_path"] = str(augmented_path)
    augmented_eval["pass_threshold"] = pass_threshold

    baseline_out = run_dir / str(gcfg["output"]["baseline_rerun_json"])
    augmented_out = run_dir / str(gcfg["output"]["augmented_rerun_json"])
    write_json(baseline_out, baseline_eval)
    write_json(augmented_out, augmented_eval)

    print(f"BASELINE_AVG_SCORE={baseline_eval['avg_case_score']:.4f}")
    print(f"AUGMENTED_AVG_SCORE={augmented_eval['avg_case_score']:.4f}")
    print(f"BASELINE_PATH={baseline_out}")
    print(f"AUGMENTED_PATH={augmented_out}")


if __name__ == "__main__":
    main()

