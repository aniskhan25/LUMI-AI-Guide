#!/usr/bin/env python3
"""Validate Lesson 04 evaluation artifacts."""

import argparse
import json
from pathlib import Path

from _common import load_config, read_jsonl, resolve_path, resolve_run_dir


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
    cfg = load_config(args.config)
    run_root = resolve_run_dir(cfg, args.config)

    eval_rows = read_jsonl(resolve_path(args.config.parent, str(cfg["paths"]["eval_set_jsonl"])))
    if not eval_rows:
        raise SystemExit("Evaluation set is empty")
    eval_ids = {str(row["query_id"]) for row in eval_rows}

    variant_sizes = {}
    gpu_counts = []
    for variant_key, variant_cfg in cfg["variants"].items():
        variant_name = str(variant_cfg["name"])
        variant_dir = run_root / variant_name
        outputs_path = variant_dir / "system_outputs.jsonl"
        scored_path = variant_dir / "scored_records.jsonl"
        summary_path = variant_dir / "summary.json"
        failure_path = variant_dir / "failure_samples.jsonl"
        failure_summary_path = variant_dir / "failure_summary.json"
        metadata_path = variant_dir / "run_metadata.json"

        for path in [outputs_path, scored_path, summary_path, failure_path, failure_summary_path, metadata_path]:
            ensure_file(path)

        outputs = read_jsonl(outputs_path)
        scored = read_jsonl(scored_path)
        output_ids = {str(row["query_id"]) for row in outputs}
        scored_ids = {str(row["query_id"]) for row in scored}
        if output_ids != eval_ids:
            raise SystemExit(f"Output IDs do not match evaluation IDs for variant {variant_name}")
        if scored_ids != eval_ids:
            raise SystemExit(f"Scored IDs do not match evaluation IDs for variant {variant_name}")

        summary = load_json(summary_path)
        if int(summary.get("item_count", -1)) != len(eval_rows):
            raise SystemExit(f"Summary item_count mismatch for variant {variant_name}")
        if int(summary.get("gpu_visible_count", 0)) < 1:
            raise SystemExit(f"Summary reports gpu_visible_count < 1 for variant {variant_name}")
        gpu_counts.append(int(summary.get("gpu_visible_count", 0)))
        variant_sizes[variant_name] = len(outputs)

    comparison_path = run_root / "comparison.json"
    report_path = run_root / "evaluation_report.md"
    ensure_file(comparison_path)
    ensure_file(report_path)

    comparison = load_json(comparison_path)
    report_text = report_path.read_text(encoding="utf-8")
    recommendation = str(comparison.get("recommendation", ""))
    if recommendation and recommendation not in report_text:
        raise SystemExit("Report does not include the comparison recommendation")

    print("VALIDATION_OK=1")
    print(f"items={len(eval_rows)}")
    print(f"baseline_items={variant_sizes[str(cfg['variants']['baseline']['name'])]}")
    print(f"candidate_items={variant_sizes[str(cfg['variants']['candidate']['name'])]}")
    print(f"gpu_visible_count={max(gpu_counts) if gpu_counts else 0}")


if __name__ == "__main__":
    main()
