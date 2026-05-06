#!/usr/bin/env python3
"""Merge accepted synthetic records with the baseline dataset."""

import argparse
import json
from pathlib import Path

from _common import load_yaml, qa_key, read_jsonl, resolve_path, resolve_run_dir, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)

    baseline_path = resolve_path(args.config.parent, str(cfg["paths"]["baseline_dataset_jsonl"]))
    accepted_path = run_dir / str(cfg["output"]["accepted_jsonl"])
    baseline_rows = read_jsonl(baseline_path)
    accepted_rows = read_jsonl(accepted_path)

    dataset_version = f"{run_dir.name}-augmented-v1"
    out_rows = []
    seen = set()
    dropped_duplicates = 0

    for idx, row in enumerate(baseline_rows):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        key = qa_key(question, answer)
        if key in seen:
            dropped_duplicates += 1
            continue
        seen.add(key)
        out_rows.append(
            {
                "record_id": str(row.get("record_id", f"base-{idx + 1:04d}")),
                "question": question,
                "answer": answer,
                "gap_label": str(row.get("gap_label", "general")),
                "source_flag": "original",
                "dataset_version": dataset_version,
            }
        )

    for row in accepted_rows:
        question = str(row.get("generated_input", "")).strip()
        answer = str(row.get("generated_target", "")).strip()
        key = qa_key(question, answer)
        if key in seen:
            dropped_duplicates += 1
            continue
        seen.add(key)
        synthetic_id = str(row.get("synthetic_id", ""))
        out_rows.append(
            {
                "record_id": f"syn-{synthetic_id}",
                "question": question,
                "answer": answer,
                "gap_label": str(row.get("gap_label", "general")),
                "source_flag": "synthetic",
                "dataset_version": dataset_version,
                "synthetic_id": synthetic_id,
                "source_case_id": str(row.get("source_case_id", "")),
                "provenance": row.get("provenance", {}),
            }
        )

    augmented_path = run_dir / str(cfg["output"]["augmented_dataset_jsonl"])
    write_jsonl(augmented_path, out_rows)

    summary = {
        "baseline_count": len(baseline_rows),
        "accepted_synthetic_count": len(accepted_rows),
        "augmented_count": len(out_rows),
        "dropped_duplicates": dropped_duplicates,
        "dataset_version": dataset_version,
        "augmented_dataset_path": str(augmented_path),
    }
    summary_path = run_dir / str(cfg["output"]["merge_summary_json"])
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"BASELINE_COUNT={len(baseline_rows)}")
    print(f"ACCEPTED_SYNTHETIC={len(accepted_rows)}")
    print(f"AUGMENTED_COUNT={len(out_rows)}")
    print(f"AUGMENTED_PATH={augmented_path}")


if __name__ == "__main__":
    main()
