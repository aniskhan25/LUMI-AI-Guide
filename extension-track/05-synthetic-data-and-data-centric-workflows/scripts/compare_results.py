#!/usr/bin/env python3
"""Compare baseline and augmented rerun results and build a report."""

import argparse
import json
from pathlib import Path

from _common import load_yaml, read_jsonl, resolve_run_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def weighted_score(result, weights):
    return (
        float(result.get("avg_case_score", 0.0)) * float(weights["avg_case_score"])
        + float(result.get("coverage_rate", 0.0)) * float(weights["coverage_rate"])
    )


def top_cases(cases, improve, n=3):
    return sorted(cases, key=lambda row: float(row.get("score_delta", 0.0)), reverse=improve)[:n]


def interpret_result(avg_delta, coverage_delta, accepted_count, rejected_count, recommendation):
    if recommendation == "use_augmented" and avg_delta > 0 and coverage_delta >= 0:
        return [
            "- Interpretation: `acceptable`",
            "- Why: the augmented dataset improved the measured weak cases without losing coverage.",
        ]
    if recommendation == "use_augmented" and coverage_delta < 0:
        return [
            "- Interpretation: `risky`",
            "- Why: the recommendation favors augmentation, but coverage regressed and should be reviewed before adoption.",
        ]
    if accepted_count == 0:
        return [
            "- Interpretation: `inconclusive`",
            "- Why: no synthetic candidates were accepted, so the loop did not really test augmentation.",
        ]
    return [
        "- Interpretation: `keep baseline`",
        "- Why: the augmented dataset did not improve the weak cases enough to justify adoption.",
    ]


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)

    baseline = load_json(run_dir / str(cfg["output"]["baseline_rerun_json"]))
    augmented = load_json(run_dir / str(cfg["output"]["augmented_rerun_json"]))
    filter_summary = load_json(run_dir / str(cfg["output"]["filter_summary_json"]))
    accepted = read_jsonl(run_dir / str(cfg["output"]["accepted_jsonl"]))
    rejected = read_jsonl(run_dir / str(cfg["output"]["rejected_jsonl"]))

    baseline_weighted = weighted_score(baseline, cfg["comparison"]["weighted_score"])
    augmented_weighted = weighted_score(augmented, cfg["comparison"]["weighted_score"])
    weighted_delta = augmented_weighted - baseline_weighted
    coverage_delta = float(augmented["coverage_rate"]) - float(baseline["coverage_rate"])
    avg_delta = float(augmented["avg_case_score"]) - float(baseline["avg_case_score"])

    min_score_delta = float(cfg["comparison"]["min_expected_score_delta"])
    min_coverage_delta = float(cfg["comparison"]["min_expected_coverage_delta"])

    recommend_augmented = avg_delta >= min_score_delta and coverage_delta >= min_coverage_delta
    if bool(cfg["comparison"]["prefer_augmented_if_no_regression"]) and avg_delta >= 0 and coverage_delta >= 0:
        recommend_augmented = True

    baseline_by_case = {str(row["case_id"]): row for row in baseline["per_case"]}
    case_deltas = []
    for row in augmented["per_case"]:
        case_id = str(row["case_id"])
        baseline_row = baseline_by_case.get(case_id, {})
        case_deltas.append(
            {
                "case_id": case_id,
                "baseline_score": float(baseline_row.get("best_score", 0.0)),
                "augmented_score": float(row.get("best_score", 0.0)),
                "score_delta": float(row.get("best_score", 0.0)) - float(baseline_row.get("best_score", 0.0)),
            }
        )

    recommendation = "use_augmented" if recommend_augmented else "keep_baseline"
    comparison = {
        "baseline_avg_case_score": baseline["avg_case_score"],
        "augmented_avg_case_score": augmented["avg_case_score"],
        "baseline_coverage_rate": baseline["coverage_rate"],
        "augmented_coverage_rate": augmented["coverage_rate"],
        "avg_case_score_delta": avg_delta,
        "coverage_rate_delta": coverage_delta,
        "baseline_weighted_score": baseline_weighted,
        "augmented_weighted_score": augmented_weighted,
        "weighted_score_delta": weighted_delta,
        "accepted_count": filter_summary["accepted_count"],
        "rejected_count": filter_summary["rejected_count"],
        "recommendation": recommendation,
        "case_deltas": case_deltas,
    }

    comparison_path = run_dir / str(cfg["output"]["comparison_json"])
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    improved = top_cases(case_deltas, improve=True)
    regressed = top_cases(case_deltas, improve=False)

    report_lines = [
        "# Synthetic Data Comparison Report",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Recommendation: `{comparison['recommendation']}`",
        "",
        "## Metric Summary",
        "",
        "| Metric | Baseline | Augmented | Delta |",
        "|---|---:|---:|---:|",
        f"| avg_case_score | {baseline['avg_case_score']:.4f} | {augmented['avg_case_score']:.4f} | {avg_delta:+.4f} |",
        f"| coverage_rate | {baseline['coverage_rate']:.4f} | {augmented['coverage_rate']:.4f} | {coverage_delta:+.4f} |",
        "",
        "## Interpretation",
        "",
    ]
    report_lines.extend(
        interpret_result(
            avg_delta,
            coverage_delta,
            int(filter_summary["accepted_count"]),
            int(filter_summary["rejected_count"]),
            recommendation,
        )
    )
    report_lines.extend(
        [
            "",
            "## Data Curation Summary",
            "",
            f"- candidates accepted: `{filter_summary['accepted_count']}`",
            f"- candidates rejected: `{filter_summary['rejected_count']}`",
            f"- acceptance rate: `{filter_summary['acceptance_rate']:.4f}`",
            "",
            "## Most Improved Cases",
        ]
    )
    for row in improved:
        report_lines.append(f"- `{row['case_id']}` delta={row['score_delta']:+.4f}")
    report_lines.append("")
    report_lines.append("## Most Regressed Cases")
    for row in regressed:
        report_lines.append(f"- `{row['case_id']}` delta={row['score_delta']:+.4f}")
    report_lines.extend(
        [
            "",
            "## Manual Inspection Samples",
            "",
        ]
    )
    if accepted:
        report_lines.append("- Accepted sample:")
        report_lines.append(f"  - `{accepted[0].get('synthetic_id', '')}`: {str(accepted[0].get('generated_input', ''))[:140]}")
    if rejected:
        report_lines.append("- Rejected sample:")
        report_lines.append(
            f"  - `{rejected[0].get('synthetic_id', '')}` reasons={rejected[0].get('filter_reasons', [])}"
        )
    report_lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Do not prefer augmentation just because many candidates were generated or accepted.",
            "- Prefer the baseline if the accepted synthetic records do not improve the intended weak cases.",
            "- Keep this report with the saved synthetic-data artifacts.",
        ]
    )

    report_path = run_dir / str(cfg["output"]["comparison_report_md"])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"COMPARISON_PATH={comparison_path}")
    print(f"REPORT_PATH={report_path}")
    print(f"RECOMMENDATION={comparison['recommendation']}")


if __name__ == "__main__":
    main()
