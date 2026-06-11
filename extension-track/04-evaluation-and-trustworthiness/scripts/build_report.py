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


def improvement_lines(baseline, candidate):
    lines = []
    metric_names = [
        "retrieval_hit_rate",
        "answer_score_mean",
        "grounded_rate",
        "completion_rate",
        "pass_rate",
    ]
    for name in metric_names:
        baseline_value = float(baseline.get(name, 0.0))
        candidate_value = float(candidate.get(name, 0.0))
        if candidate_value > baseline_value:
            lines.append(f"- `{name}` improved by `{candidate_value - baseline_value:.4f}`.")
        elif candidate_value < baseline_value:
            lines.append(f"- `{name}` worsened by `{baseline_value - candidate_value:.4f}`.")
    if not lines:
        lines.append("- The aggregate metrics were effectively unchanged.")
    return lines


def interpret_result(baseline, candidate, comparison, baseline_failures, candidate_failures):
    unsupported_baseline = sum(1 for row in baseline_failures if row.get("failure_category") == "answer_unsupported_by_evidence")
    unsupported_candidate = sum(1 for row in candidate_failures if row.get("failure_category") == "answer_unsupported_by_evidence")
    weighted_delta = float(comparison["candidate_weighted_score"]) - float(comparison["baseline_weighted_score"])
    grounded_delta = float(candidate.get("grounded_rate", 0.0)) - float(baseline.get("grounded_rate", 0.0))
    answer_delta = float(candidate.get("answer_score_mean", 0.0)) - float(baseline.get("answer_score_mean", 0.0))

    if weighted_delta > 0 and grounded_delta >= 0 and unsupported_candidate <= unsupported_baseline:
        status = "acceptable"
        reason = "The candidate improved or preserved the important metrics without introducing more unsupported-answer failures in the sampled set."
    elif weighted_delta > 0 and (grounded_delta < 0 or unsupported_candidate > unsupported_baseline):
        status = "risky"
        reason = "The candidate improved the weighted score, but grounding or unsupported-answer behavior got worse."
    elif weighted_delta <= 0 and answer_delta <= 0:
        status = "keep baseline"
        reason = "The candidate did not improve the overall result enough to justify adoption."
    else:
        status = "inconclusive"
        reason = "The aggregate changes are mixed, so the failure samples should drive the next decision."

    return [
        f"- Interpretation: `{status}`",
        f"- Why: {reason}",
    ]


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
        "## Interpretation",
        "",
    ]
    lines.extend(interpret_result(baseline_summary, candidate_summary, comparison, baseline_failures, candidate_failures))
    lines.extend(
        [
            "",
            "### Metric Reading",
            "",
        ]
    )
    lines.extend(improvement_lines(baseline_summary, candidate_summary))
    lines.extend(
        [
            "",
        "## Failure Samples",
        "",
        ]
    )
    lines.extend(format_failure_samples(baseline_failures, f"Baseline ({baseline_name})"))
    lines.append("")
    lines.extend(format_failure_samples(candidate_failures, f"Candidate ({candidate_name})"))
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Do not adopt the candidate for a small score gain if the failures become less grounded or harder to explain.",
            "- Prefer the variant whose failures are safer in the categories that matter most to the task.",
            "- Keep this report with the saved evaluation artifacts.",
        ]
    )

    report_path = run_root / "evaluation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
