#!/usr/bin/env python3
"""Compare baseline and candidate summaries."""

import argparse
import json
from pathlib import Path

from _common import load_config, resolve_run_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_summary(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def weighted(summary, weights):
    score = 0.0
    for metric, weight in weights.items():
        score += float(summary.get(metric, 0.0)) * float(weight)
    return score


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_root = resolve_run_dir(cfg, args.config)

    baseline_key = str(cfg["comparison"]["baseline_variant"])
    candidate_key = str(cfg["comparison"]["candidate_variant"])
    baseline_name = str(cfg["variants"][baseline_key]["name"])
    candidate_name = str(cfg["variants"][candidate_key]["name"])

    baseline_summary = load_summary(run_root / baseline_name / "summary.json")
    candidate_summary = load_summary(run_root / candidate_name / "summary.json")

    weights = cfg["comparison"]["weighted_score"]
    baseline_weighted = weighted(baseline_summary, weights)
    candidate_weighted = weighted(candidate_summary, weights)

    deltas = {
        "retrieval_hit_rate": float(candidate_summary["retrieval_hit_rate"]) - float(baseline_summary["retrieval_hit_rate"]),
        "answer_score_mean": float(candidate_summary["answer_score_mean"]) - float(baseline_summary["answer_score_mean"]),
        "grounded_rate": float(candidate_summary["grounded_rate"]) - float(baseline_summary["grounded_rate"]),
        "completion_rate": float(candidate_summary["completion_rate"]) - float(baseline_summary["completion_rate"]),
        "pass_rate": float(candidate_summary["pass_rate"]) - float(baseline_summary["pass_rate"]),
        "weighted_score": candidate_weighted - baseline_weighted,
    }

    recommendation = candidate_name if candidate_weighted >= baseline_weighted else baseline_name
    comparison = {
        "baseline_variant": baseline_name,
        "candidate_variant": candidate_name,
        "baseline_weighted_score": baseline_weighted,
        "candidate_weighted_score": candidate_weighted,
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
