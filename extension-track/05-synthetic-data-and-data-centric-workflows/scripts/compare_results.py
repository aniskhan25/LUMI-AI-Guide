#!/usr/bin/env python3
"""Compare baseline vs augmented rerun results and build report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _common import load_yaml, read_jsonl, resolve_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-config", type=Path, required=True)
    parser.add_argument("--compare-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def weighted_score(result: Dict[str, Any], weights: Dict[str, float]) -> float:
    return (
        float(result.get("avg_case_score", 0.0)) * float(weights["avg_case_score"])
        + float(result.get("coverage_rate", 0.0)) * float(weights["coverage_rate"])
    )


def top_cases(cases: List[Dict[str, Any]], improve: bool, n: int = 3) -> List[Dict[str, Any]]:
    return sorted(cases, key=lambda x: float(x.get("score_delta", 0.0)), reverse=improve)[:n]


def main() -> None:
    args = parse_args()
    gcfg = load_yaml(args.generate_config)
    ccfg = load_yaml(args.compare_config)
    run_dir = resolve_run_dir(gcfg, args.generate_config, args.output_root, args.run_name)

    baseline = load_json(run_dir / str(gcfg["output"]["baseline_rerun_json"]))
    augmented = load_json(run_dir / str(gcfg["output"]["augmented_rerun_json"]))
    filter_summary = load_json(run_dir / str(gcfg["output"]["filter_summary_json"]))
    accepted = read_jsonl(run_dir / str(gcfg["output"]["accepted_jsonl"]))
    rejected = read_jsonl(run_dir / str(gcfg["output"]["rejected_jsonl"]))

    b_score = weighted_score(baseline, ccfg["decision"]["weighted_score"])
    a_score = weighted_score(augmented, ccfg["decision"]["weighted_score"])
    score_delta = a_score - b_score
    coverage_delta = float(augmented["coverage_rate"]) - float(baseline["coverage_rate"])
    avg_delta = float(augmented["avg_case_score"]) - float(baseline["avg_case_score"])

    min_score_delta = float(ccfg["evaluation"]["min_expected_score_delta"])
    min_cov_delta = float(ccfg["evaluation"]["min_expected_coverage_delta"])

    recommend_augmented = avg_delta >= min_score_delta and coverage_delta >= min_cov_delta
    if bool(ccfg["decision"]["prefer_augmented_if_no_regression"]) and avg_delta >= 0 and coverage_delta >= 0:
        recommend_augmented = True

    baseline_by_case = {str(x["case_id"]): x for x in baseline["per_case"]}
    case_deltas: List[Dict[str, Any]] = []
    for row in augmented["per_case"]:
        case_id = str(row["case_id"])
        b = baseline_by_case.get(case_id, {})
        case_deltas.append(
            {
                "case_id": case_id,
                "baseline_score": float(b.get("best_score", 0.0)),
                "augmented_score": float(row.get("best_score", 0.0)),
                "score_delta": float(row.get("best_score", 0.0)) - float(b.get("best_score", 0.0)),
            }
        )

    comparison = {
        "baseline_avg_case_score": baseline["avg_case_score"],
        "augmented_avg_case_score": augmented["avg_case_score"],
        "baseline_coverage_rate": baseline["coverage_rate"],
        "augmented_coverage_rate": augmented["coverage_rate"],
        "avg_case_score_delta": avg_delta,
        "coverage_rate_delta": coverage_delta,
        "baseline_weighted_score": b_score,
        "augmented_weighted_score": a_score,
        "weighted_score_delta": score_delta,
        "accepted_count": filter_summary["accepted_count"],
        "rejected_count": filter_summary["rejected_count"],
        "recommendation": "use_augmented" if recommend_augmented else "keep_baseline",
        "case_deltas": case_deltas,
    }

    comparison_path = run_dir / str(gcfg["output"]["comparison_json"])
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    improved = top_cases(case_deltas, improve=True, n=3)
    regressed = top_cases(case_deltas, improve=False, n=3)

    report_lines: List[str] = []
    report_lines.append("# Synthetic Data Comparison Report")
    report_lines.append("")
    report_lines.append(f"- Run: `{run_dir.name}`")
    report_lines.append(f"- Recommendation: `{comparison['recommendation']}`")
    report_lines.append("")
    report_lines.append("## Metric Summary")
    report_lines.append("")
    report_lines.append("| Metric | Baseline | Augmented | Delta |")
    report_lines.append("|---|---:|---:|---:|")
    report_lines.append(f"| avg_case_score | {baseline['avg_case_score']:.4f} | {augmented['avg_case_score']:.4f} | {avg_delta:+.4f} |")
    report_lines.append(f"| coverage_rate | {baseline['coverage_rate']:.4f} | {augmented['coverage_rate']:.4f} | {coverage_delta:+.4f} |")
    report_lines.append("")
    report_lines.append("## Data Curation Summary")
    report_lines.append("")
    report_lines.append(f"- candidates accepted: `{filter_summary['accepted_count']}`")
    report_lines.append(f"- candidates rejected: `{filter_summary['rejected_count']}`")
    report_lines.append(f"- acceptance rate: `{filter_summary['acceptance_rate']:.4f}`")
    report_lines.append("")
    report_lines.append("## Most Improved Cases")
    for row in improved:
        report_lines.append(f"- `{row['case_id']}` delta={row['score_delta']:+.4f}")
    report_lines.append("")
    report_lines.append("## Most Regressed Cases")
    for row in regressed:
        report_lines.append(f"- `{row['case_id']}` delta={row['score_delta']:+.4f}")
    report_lines.append("")
    report_lines.append("## Manual Inspection Samples")
    report_lines.append("")
    if accepted:
        report_lines.append("- Accepted sample:")
        report_lines.append(f"  - `{accepted[0].get('synthetic_id','')}`: {str(accepted[0].get('generated_input',''))[:140]}")
    if rejected:
        report_lines.append("- Rejected sample:")
        report_lines.append(
            f"  - `{rejected[0].get('synthetic_id','')}` reasons={rejected[0].get('filter_reasons', [])}"
        )

    report_path = run_dir / str(gcfg["output"]["comparison_report_md"])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"COMPARISON_PATH={comparison_path}")
    print(f"REPORT_PATH={report_path}")
    print(f"RECOMMENDATION={comparison['recommendation']}")


if __name__ == "__main__":
    main()

