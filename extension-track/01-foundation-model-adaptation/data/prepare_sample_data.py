#!/usr/bin/env python3
"""Generate a small JSONL dataset for lesson smoke testing."""

import argparse
import json
import random
from pathlib import Path


NEGATIVE_TEXTS = [
    "service timeout while loading model weights",
    "request queue overflow during peak traffic",
    "invalid token caused failed authentication",
    "database connection dropped unexpectedly",
    "worker crashed before checkpoint save",
]

POSITIVE_TEXTS = [
    "model served response within latency target",
    "training job completed with stable loss",
    "new deployment passed all health checks",
    "batch inference completed without errors",
    "evaluation metrics improved on validation set",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--eval-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_record(label, rnd):
    base = rnd.choice(POSITIVE_TEXTS if label == 1 else NEGATIVE_TEXTS)
    noise = rnd.choice(["", " for region A", " at midnight", " in production", " on node 3"])
    return {"text": f"{base}{noise}".strip(), "label": label}


def write_jsonl(path, items):
    with path.open("w", encoding="utf-8") as f:
        for row in items:
            f.write(json.dumps(row) + "\n")


def build_split(size, rnd):
    out = []
    for _ in range(size):
        label = rnd.randint(0, 1)
        out.append(make_record(label, rnd))
    rnd.shuffle(out)
    return out


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(args.seed)

    train_rows = build_split(args.train_size, rnd)
    eval_rows = build_split(args.eval_size, rnd)

    train_path = args.output / "train.jsonl"
    eval_path = args.output / "eval.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    print(f"WROTE={train_path}")
    print(f"WROTE={eval_path}")


if __name__ == "__main__":
    main()
