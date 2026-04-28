#!/usr/bin/env python3
"""Validate inference outputs for Lesson 02."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["embeddings", "generation"], required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_ids(input_rows, output_rows):
    in_ids = [str(r["id"]) for r in input_rows]
    out_ids = [str(r["id"]) for r in output_rows]
    if len(in_ids) != len(out_ids):
        raise SystemExit(f"Count mismatch: input={len(in_ids)} output={len(out_ids)}")
    if set(in_ids) != set(out_ids):
        raise SystemExit("ID mismatch between input and output")


def validate_embeddings(output_rows):
    if not output_rows:
        raise SystemExit("No output rows found")
    dim = len(output_rows[0].get("embedding", []))
    if dim == 0:
        raise SystemExit("First embedding missing or empty")
    for i, row in enumerate(output_rows):
        emb = row.get("embedding")
        if not isinstance(emb, list) or len(emb) != dim:
            raise SystemExit(f"Inconsistent embedding dimension at row {i}")
    return dim


def validate_generation(output_rows):
    if not output_rows:
        raise SystemExit("No output rows found")
    for i, row in enumerate(output_rows):
        text = row.get("output_text")
        if not isinstance(text, str) or not text.strip():
            raise SystemExit(f"Empty generated output at row {i}")


def main():
    args = parse_args()
    input_rows = read_jsonl(args.input_jsonl)
    output_rows = read_jsonl(args.output_jsonl)
    if not input_rows:
        raise SystemExit("Input JSONL is empty")

    validate_ids(input_rows, output_rows)

    if args.mode == "embeddings":
        embedding_dim = validate_embeddings(output_rows)
    else:
        validate_generation(output_rows)
        embedding_dim = None

    if not args.summary_json.is_file():
        raise SystemExit(f"Summary file not found: {args.summary_json}")
    with args.summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    if int(summary.get("records_written", -1)) != len(output_rows):
        raise SystemExit(
            f"Summary records mismatch: summary={summary.get('records_written')} output={len(output_rows)}"
        )
    if int(summary.get("gpu_visible_count", 0)) < 1:
        raise SystemExit("Summary indicates no visible GPU")

    print("VALIDATION_OK=1")
    print(f"mode={args.mode}")
    print(f"input_records={len(input_rows)}")
    print(f"output_records={len(output_rows)}")
    if embedding_dim is not None:
        print(f"embedding_dim={embedding_dim}")


if __name__ == "__main__":
    main()
