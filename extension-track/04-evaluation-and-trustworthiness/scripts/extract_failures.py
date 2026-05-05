#!/usr/bin/env python3
"""Extract representative failure cases for one variant."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _common import load_config, read_jsonl, resolve_path, resolve_run_dir, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--max-per-category", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_root = resolve_run_dir(cfg, args.config)

    if args.variant not in cfg["variants"]:
        raise SystemExit(f"Unknown variant: {args.variant}")

    variant_name = str(cfg["variants"][args.variant]["name"])
    variant_dir = run_root / variant_name

    scored_path = variant_dir / "scored_records.jsonl"
    outputs_path = variant_dir / "system_outputs.jsonl"
    eval_set_path = resolve_path(args.config.parent, str(cfg["paths"]["eval_set_jsonl"]))
    ref_path = resolve_path(args.config.parent, str(cfg["paths"]["reference_answers_jsonl"]))

    scored = read_jsonl(scored_path)
    outputs = read_jsonl(outputs_path)
    eval_rows = read_jsonl(eval_set_path)
    ref_rows = read_jsonl(ref_path)

    out_by_id = {str(row["query_id"]): row for row in outputs}
    eval_by_id = {str(row["query_id"]): row for row in eval_rows}
    ref_by_id = {str(row["query_id"]): row for row in ref_rows}

    by_category = defaultdict(list)
    for row in scored:
        category = str(row.get("failure_category", "pass"))
        if category == "pass":
            continue
        by_category[category].append(row)

    selected = []
    for category, rows in by_category.items():
        rows_sorted = sorted(rows, key=lambda row: float(row.get("answer_score", 0.0)))
        for row in rows_sorted[: args.max_per_category]:
            query_id = str(row["query_id"])
            selected.append(
                {
                    "query_id": query_id,
                    "variant": variant_name,
                    "failure_category": category,
                    "question": eval_by_id.get(query_id, {}).get("question", ""),
                    "expected_doc_id": eval_by_id.get(query_id, {}).get("expected_doc_id", ""),
                    "reference_answer": ref_by_id.get(query_id, {}).get("reference_answer", ""),
                    "answer": out_by_id.get(query_id, {}).get("answer", ""),
                    "evidence_chunk_ids": out_by_id.get(query_id, {}).get("evidence_chunk_ids", []),
                    "answer_score": row.get("answer_score", 0.0),
                }
            )

    failure_path = variant_dir / "failure_samples.jsonl"
    write_jsonl(failure_path, selected)

    summary = {
        "variant": variant_name,
        "total_failures": sum(len(rows) for rows in by_category.values()),
        "sampled_failures": len(selected),
        "categories": {category: len(rows) for category, rows in by_category.items()},
    }
    summary_path = variant_dir / "failure_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"VARIANT={variant_name}")
    print(f"FAILURE_SAMPLES={failure_path}")
    print(f"FAILURE_SUMMARY={summary_path}")
    print(f"SAMPLED={len(selected)}")


if __name__ == "__main__":
    main()
