#!/usr/bin/env python3
"""Build markdown summary report from evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _common import load_config, read_jsonl, resolve_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_row(name: str, baseline: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    b = float(baseline.get(name, 0.0))
    c = float(candidate.get(name, 0.0))
    d = c - b
    return f"| {name} | {b:.4f} | {c:.4f} | {d:+.4f} |"


def format_failure_samples(rows: List[Dict[str, Any]], title: str) -> List[str]:
    lines = [f"### {title}"]
    if not rows:
        lines.append("- No sampled failures.")
        return lines
    for row in rows[:3]:
        lines.append(f"- `{row.get('query_id')}` [{row.get('failure_category')}]: {row.get('question')}")
        lines.append(f"  - answer: {row.get('answer', '')[:160]}")
    return lines


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = resolve_run_dir(cfg, args.config, args.output_root, args.run_name)

    baseline_key = str(cfg["comparison"]["baseline_variant"])
    candidate_key = str(cfg["comparison"]["candidate_variant"])
    baseline_name = str(cfg["variants"][baseline_key]["name"])
    candidate_name = str(cfg["variants"][candidate_key]["name"])

    baseline_summary = load_json(run_root / baseline_name / "summary.json")
    candidate_summary = load_json(run_root / candidate_name / "summary.json")
    comparison = load_json(run_root / "comparison.json")

    baseline_failures = read_jsonl(run_root / baseline_name / "failure_samples.jsonl")
    candidate_failures = read_jsonl(run_root / candidate_name / "failure_samples.jsonl")

    lines: List[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"- Run: `{run_root.name}`")
    lines.append(f"- Baseline variant: `{baseline_name}`")
    lines.append(f"- Candidate variant: `{candidate_name}`")
    lines.append("")
    lines.append("## Metric Comparison")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(metric_row("retrieval_hit_rate", baseline_summary, candidate_summary))
    lines.append(metric_row("answer_score_mean", baseline_summary, candidate_summary))
    lines.append(metric_row("grounded_rate", baseline_summary, candidate_summary))
    lines.append(metric_row("completion_rate", baseline_summary, candidate_summary))
    lines.append(metric_row("pass_rate", baseline_summary, candidate_summary))
    lines.append("")
    lines.append("## Weighted Decision")
    lines.append("")
    lines.append(f"- Baseline weighted score: `{comparison['baseline_weighted_score']:.4f}`")
    lines.append(f"- Candidate weighted score: `{comparison['candidate_weighted_score']:.4f}`")
    lines.append(f"- Recommendation: `{comparison['recommendation']}`")
    lines.append("")
    lines.append("## Failure Samples")
    lines.append("")
    lines.extend(format_failure_samples(baseline_failures, f"Baseline ({baseline_name})"))
    lines.append("")
    lines.extend(format_failure_samples(candidate_failures, f"Candidate ({candidate_name})"))
    lines.append("")
    lines.append("## Decision Notes")
    lines.append("")
    lines.append("- Confirm recommendation aligns with customer risk tolerance.")
    lines.append("- Review high-impact failure categories before deployment decisions.")
    lines.append("- Keep this report with versioned evaluation artifacts for traceability.")

    report_path = run_root / "evaluation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()

