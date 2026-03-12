#!/usr/bin/env python3
"""Merge accepted synthetic records with baseline dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _common import load_yaml, qa_key, read_jsonl, resolve_path, resolve_run_dir, write_jsonl


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
    _ = load_yaml(args.filter_config)
    run_dir = resolve_run_dir(gcfg, args.generate_config, args.output_root, args.run_name)

    baseline_path = resolve_path(args.generate_config.parent, str(gcfg["paths"]["baseline_dataset_jsonl"]))
    accepted_path = run_dir / str(gcfg["output"]["accepted_jsonl"])

    baseline_rows = read_jsonl(baseline_path)
    accepted_rows = read_jsonl(accepted_path)

    dataset_version = f"{run_dir.name}-augmented-v1"
    out_rows: List[Dict[str, Any]] = []
    seen = set()
    dropped_duplicates = 0

    for i, row in enumerate(baseline_rows):
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        key = qa_key(q, a)
        if key in seen:
            dropped_duplicates += 1
            continue
        seen.add(key)
        out_rows.append(
            {
                "record_id": str(row.get("record_id", f"base-{i+1:04d}")),
                "question": q,
                "answer": a,
                "gap_label": str(row.get("gap_label", "general")),
                "source_flag": "original",
                "dataset_version": dataset_version,
            }
        )

    for row in accepted_rows:
        q = str(row.get("generated_input", "")).strip()
        a = str(row.get("generated_target", "")).strip()
        key = qa_key(q, a)
        if key in seen:
            dropped_duplicates += 1
            continue
        seen.add(key)
        syn_id = str(row.get("synthetic_id", ""))
        out_rows.append(
            {
                "record_id": f"syn-{syn_id}",
                "question": q,
                "answer": a,
                "gap_label": str(row.get("gap_label", "general")),
                "source_flag": "synthetic",
                "dataset_version": dataset_version,
                "synthetic_id": syn_id,
                "source_case_id": str(row.get("source_case_id", "")),
                "provenance": row.get("provenance", {}),
            }
        )

    augmented_path = run_dir / str(gcfg["output"]["augmented_dataset_jsonl"])
    write_jsonl(augmented_path, out_rows)

    summary = {
        "baseline_count": len(baseline_rows),
        "accepted_synthetic_count": len(accepted_rows),
        "augmented_count": len(out_rows),
        "dropped_duplicates": dropped_duplicates,
        "dataset_version": dataset_version,
        "augmented_dataset_path": str(augmented_path),
    }
    summary_path = run_dir / str(gcfg["output"]["merge_summary_json"])
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"BASELINE_COUNT={len(baseline_rows)}")
    print(f"ACCEPTED_SYNTHETIC={len(accepted_rows)}")
    print(f"AUGMENTED_COUNT={len(out_rows)}")
    print(f"AUGMENTED_PATH={augmented_path}")


if __name__ == "__main__":
    main()

