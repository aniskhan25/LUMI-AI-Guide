#!/usr/bin/env python3
"""Validate expected outputs from the Lesson 01 adaptation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing expected file: {path}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    summary_path = run_dir / "run_summary.json"
    metrics_path = run_dir / "metrics.json"
    checkpoint_dir = run_dir / "checkpoint"

    require_file(summary_path)
    require_file(metrics_path)
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Missing checkpoint directory: {checkpoint_dir}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    gpu_visible_count = int(summary.get("gpu_visible_count", 0))
    if gpu_visible_count < 1 and not args.allow_cpu:
        raise SystemExit(
            "Validation failed: gpu_visible_count < 1. Pass --allow-cpu only for non-LUMI local checks."
        )

    eval_acc = float(metrics.get("eval_accuracy", 0.0))
    if eval_acc < args.min_accuracy:
        raise SystemExit(
            f"Validation failed: eval_accuracy={eval_acc:.4f} below min-accuracy={args.min_accuracy:.4f}"
        )

    print(f"VALIDATION_OK=1 run_dir={run_dir}")
    print(f"gpu_visible_count={gpu_visible_count}")
    print(f"eval_accuracy={eval_acc:.6f}")


if __name__ == "__main__":
    main()

