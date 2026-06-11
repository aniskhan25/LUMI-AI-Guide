#!/usr/bin/env python3
"""Validate Lesson 05 synthetic-data artifacts."""

import argparse
import json
from pathlib import Path

from _common import load_yaml, read_jsonl, resolve_path, resolve_run_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def ensure_file(path):
    if not path.is_file():
        raise SystemExit(f"Missing expected file: {path}")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)

    baseline_path = resolve_path(args.config.parent, str(cfg["paths"]["baseline_dataset_jsonl"]))
    ensure_file(baseline_path)

    selected_path = run_dir / str(cfg["output"]["selected_weak_cases_jsonl"])
    selection_summary_path = run_dir / str(cfg["output"]["weak_case_selection_summary_json"])
    candidates_path = run_dir / str(cfg["output"]["candidates_jsonl"])
    generation_summary_path = run_dir / str(cfg["output"]["generation_summary_json"])
    filtered_path = run_dir / str(cfg["output"]["filter_all_jsonl"])
    accepted_path = run_dir / str(cfg["output"]["accepted_jsonl"])
    rejected_path = run_dir / str(cfg["output"]["rejected_jsonl"])
    filter_summary_path = run_dir / str(cfg["output"]["filter_summary_json"])
    augmented_path = run_dir / str(cfg["output"]["augmented_dataset_jsonl"])
    merge_summary_path = run_dir / str(cfg["output"]["merge_summary_json"])
    baseline_rerun_path = run_dir / str(cfg["output"]["baseline_rerun_json"])
    augmented_rerun_path = run_dir / str(cfg["output"]["augmented_rerun_json"])
    comparison_path = run_dir / str(cfg["output"]["comparison_json"])
    report_path = run_dir / str(cfg["output"]["comparison_report_md"])

    for path in [
        selected_path,
        selection_summary_path,
        candidates_path,
        generation_summary_path,
        filtered_path,
        accepted_path,
        rejected_path,
        filter_summary_path,
        augmented_path,
        merge_summary_path,
        baseline_rerun_path,
        augmented_rerun_path,
        comparison_path,
        report_path,
    ]:
        ensure_file(path)

    selected = read_jsonl(selected_path)
    candidates = read_jsonl(candidates_path)
    filtered = read_jsonl(filtered_path)
    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    augmented = read_jsonl(augmented_path)
    baseline_rows = read_jsonl(baseline_path)

    if not selected:
        raise SystemExit("Selected weak cases are empty")
    if not candidates:
        raise SystemExit("Synthetic candidate set is empty")
    if len(filtered) != len(candidates):
        raise SystemExit("Filtered candidate count does not match candidate count")
    if len(accepted) + len(rejected) != len(candidates):
        raise SystemExit("Accepted and rejected counts do not match candidate count")

    accepted_ids = set()
    for row in accepted:
        synthetic_id = str(row.get("synthetic_id", ""))
        source_case_id = str(row.get("source_case_id", ""))
        if not synthetic_id:
            raise SystemExit("Accepted synthetic record is missing synthetic_id")
        if not source_case_id:
            raise SystemExit("Accepted synthetic record is missing source_case_id")
        accepted_ids.add(synthetic_id)

    merge_summary = load_json(merge_summary_path)
    expected_augmented_count = (
        int(merge_summary["baseline_count"])
        + int(merge_summary["accepted_synthetic_count"])
        - int(merge_summary["dropped_duplicates"])
    )
    if len(augmented) != expected_augmented_count:
        raise SystemExit("Augmented dataset count does not match merge summary")

    synthetic_rows = [row for row in augmented if str(row.get("source_flag", "")) == "synthetic"]
    for row in synthetic_rows:
        if str(row.get("synthetic_id", "")) not in accepted_ids:
            raise SystemExit("Augmented dataset contains synthetic_id not present in accepted candidates")

    baseline_rerun = load_json(baseline_rerun_path)
    augmented_rerun = load_json(augmented_rerun_path)
    if int(baseline_rerun.get("case_count", -1)) != len(selected):
        raise SystemExit("Baseline rerun case_count does not match selected weak cases")
    if int(augmented_rerun.get("case_count", -1)) != len(selected):
        raise SystemExit("Augmented rerun case_count does not match selected weak cases")

    comparison = load_json(comparison_path)
    report_text = report_path.read_text(encoding="utf-8")
    recommendation = str(comparison.get("recommendation", ""))
    if recommendation and recommendation not in report_text:
        raise SystemExit("Comparison recommendation is missing from the report")

    generation_summary = load_json(generation_summary_path)
    gpu_visible_count = int(generation_summary.get("gpu_visible_count", 0))

    print("VALIDATION_OK=1")
    print(f"weak_cases={len(selected)}")
    print(f"candidates={len(candidates)}")
    print(f"accepted={len(accepted)}")
    print(f"augmented_records={len(augmented)}")
    print(f"gpu_visible_count={gpu_visible_count}")


if __name__ == "__main__":
    main()
