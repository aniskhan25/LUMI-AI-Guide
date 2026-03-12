#!/usr/bin/env python3
"""Compare baseline and candidate evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from _common import load_config, resolve_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def weighted(summary: Dict[str, float], weights: Dict[str, float]) -> float:
    score = 0.0
    for metric, weight in weights.items():
        score += float(summary.get(metric, 0.0)) * float(weight)
    return score


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = resolve_run_dir(cfg, args.config, args.output_root, args.run_name)

    baseline_key = str(cfg["comparison"]["baseline_variant"])
    candidate_key = str(cfg["comparison"]["candidate_variant"])
    baseline_name = str(cfg["variants"][baseline_key]["name"])
    candidate_name = str(cfg["variants"][candidate_key]["name"])

    baseline_summary = load_summary(run_root / baseline_name / "summary.json")
    candidate_summary = load_summary(run_root / candidate_name / "summary.json")

    weights = cfg["comparison"]["weighted_score"]
    b_weighted = weighted(baseline_summary, weights)
    c_weighted = weighted(candidate_summary, weights)

    deltas = {
        "retrieval_hit_rate": float(candidate_summary["retrieval_hit_rate"]) - float(baseline_summary["retrieval_hit_rate"]),
        "answer_score_mean": float(candidate_summary["answer_score_mean"]) - float(baseline_summary["answer_score_mean"]),
        "grounded_rate": float(candidate_summary["grounded_rate"]) - float(baseline_summary["grounded_rate"]),
        "completion_rate": float(candidate_summary["completion_rate"]) - float(baseline_summary["completion_rate"]),
        "pass_rate": float(candidate_summary["pass_rate"]) - float(baseline_summary["pass_rate"]),
        "weighted_score": c_weighted - b_weighted,
    }

    recommendation = candidate_name if c_weighted >= b_weighted else baseline_name

    comparison = {
        "baseline_variant": baseline_name,
        "candidate_variant": candidate_name,
        "baseline_weighted_score": b_weighted,
        "candidate_weighted_score": c_weighted,
        "deltas": deltas,
        "recommendation": recommendation,
    }
    out_path = run_root / "comparison.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"COMPARISON={out_path}")
    print(f"RECOMMENDATION={recommendation}")


if __name__ == "__main__":
    main()

