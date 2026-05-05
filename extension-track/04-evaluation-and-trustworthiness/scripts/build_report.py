#!/usr/bin/env python3
"""Build the evaluation report."""

import argparse
import json
from pathlib import Path

from _common import load_config, read_jsonl, resolve_run_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_row(name, baseline, candidate):
    baseline_value = float(baseline.get(name, 0.0))
    candidate_value = float(candidate.get(name, 0.0))
    delta = candidate_value - baseline_value
    return f"| {name} | {baseline_value:.4f} | {candidate_value:.4f} | {delta:+.4f} |"


def format_failure_samples(rows, title):
    lines = [f"### {title}"]
    if not rows:
        lines.append("- No sampled failures.")
        return lines
    for row in rows[:3]:
        lines.append(f"- `{row.get('query_id')}` [{row.get('failure_category')}]: {row.get('question')}")
        lines.append(f"  - answer: {str(row.get('answer', ''))[:160]}")
    return lines


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_root = resolve_run_dir(cfg, args.config)

    baseline_key = str(cfg["comparison"]["baseline_variant"])
    candidate_key = str(cfg["comparison"]["candidate_variant"])
    baseline_name = str(cfg["variants"][baseline_key]["name"])
    candidate_name = str(cfg["variants"][candidate_key]["name"])

    baseline_summary = load_json(run_root / baseline_name / "summary.json")
    candidate_summary = load_json(run_root / candidate_name / "summary.json")
    comparison = load_json(run_root / "comparison.json")

    baseline_failures = read_jsonl(run_root / baseline_name / "failure_samples.jsonl")
    candidate_failures = read_jsonl(run_root / candidate_name / "failure_samples.jsonl")

    lines = [
        "# Evaluation Report",
        "",
        f"- Run: `{run_root.name}`",
        f"- Baseline variant: `{baseline_name}`",
        f"- Candidate variant: `{candidate_name}`",
        "",
        "## Metric Comparison",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        metric_row("retrieval_hit_rate", baseline_summary, candidate_summary),
        metric_row("answer_score_mean", baseline_summary, candidate_summary),
        metric_row("grounded_rate", baseline_summary, candidate_summary),
        metric_row("completion_rate", baseline_summary, candidate_summary),
        metric_row("pass_rate", baseline_summary, candidate_summary),
        "",
        "## Weighted Decision",
        "",
        f"- Baseline weighted score: `{comparison['baseline_weighted_score']:.4f}`",
        f"- Candidate weighted score: `{comparison['candidate_weighted_score']:.4f}`",
        f"- Recommendation: `{comparison['recommendation']}`",
        "",
        "## Failure Samples",
        "",
    ]
    lines.extend(format_failure_samples(baseline_failures, f"Baseline ({baseline_name})"))
    lines.append("")
    lines.extend(format_failure_samples(candidate_failures, f"Candidate ({candidate_name})"))
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Confirm the recommendation matches the important customer risks.",
            "- Review high-impact failure categories before adopting the candidate.",
            "- Keep this report with the saved evaluation artifacts.",
        ]
    )

    report_path = run_root / "evaluation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
