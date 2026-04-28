#!/usr/bin/env python3
"""Download a small AG News subset and write it as JSONL."""

import argparse
import json
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit("datasets is required to prepare AG News.") from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"text": row["text"], "label": int(row["label"])}) + "\n")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("ag_news")
    train_rows = dataset["train"].shuffle(seed=args.seed).select(range(args.train_size))
    eval_rows = dataset["test"].shuffle(seed=args.seed).select(range(args.eval_size))

    train_path = args.output / "train.jsonl"
    eval_path = args.output / "eval.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    print(f"WROTE={train_path}")
    print(f"WROTE={eval_path}")


if __name__ == "__main__":
    main()
