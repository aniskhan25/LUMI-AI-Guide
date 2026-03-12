#!/usr/bin/env python3
"""Compare batched and service-style summaries and produce decision report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from _common import load_yaml, read_json, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-config", type=Path, required=True)
    return parser.parse_args()


def choose_recommendation(priority: str, batched: Dict[str, Any], service: Dict[str, Any]) -> str:
    b_t = float(batched.get("throughput_rps", 0.0))
    s_t = float(service.get("throughput_rps", 0.0))
    b_l = float(batched.get("p95_latency_ms", 0.0))
    s_l = float(service.get("p95_latency_ms", 0.0))

    if priority == "throughput_first":
        return "batched" if b_t >= s_t else "service"
    if priority == "latency_first":
        return "batched" if b_l <= s_l else "service"

    # balanced
    b_score = b_t / max(1e-9, b_l)
    s_score = s_t / max(1e-9, s_l)
    return "batched" if b_score >= s_score else "service"


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.compare_config)
    base_dir = args.compare_config.parent

    batched_summary_path = resolve_path(base_dir, str(cfg["paths"]["batched_summary"]))
    service_summary_path = resolve_path(base_dir, str(cfg["paths"]["service_summary"]))

    batched = read_json(batched_summary_path)
    service = read_json(service_summary_path)

    priority = str(cfg["comparison"]["recommendation_priority"])
    recommendation = choose_recommendation(priority, batched, service)

    deltas = {
        "throughput_rps_delta_service_minus_batched": float(service["throughput_rps"]) - float(batched["throughput_rps"]),
        "p95_latency_ms_delta_service_minus_batched": float(service["p95_latency_ms"]) - float(batched["p95_latency_ms"]),
        "completion_rate_delta_service_minus_batched": float(service["completion_rate"]) - float(batched["completion_rate"]),
    }

    payload = {
        "batched_label": str(cfg["comparison"]["batched_label"]),
        "service_label": str(cfg["comparison"]["service_label"]),
        "priority": priority,
        "batched_summary_path": str(batched_summary_path),
        "service_summary_path": str(service_summary_path),
        "deltas": deltas,
        "recommendation": recommendation,
    }

    report_json = resolve_path(base_dir, str(cfg["comparison"]["report_json"]))
    report_md = resolve_path(base_dir, str(cfg["comparison"]["report_md"]))
    write_json(report_json, payload)

    lines: List[str] = []
    lines.append("# Advanced Inference Comparison Report")
    lines.append("")
    lines.append(f"- Recommendation priority: `{priority}`")
    lines.append(f"- Recommendation: `{recommendation}`")
    lines.append("")
    lines.append("| Metric | Batched | Service | Delta (service-batched) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| throughput_rps | {float(batched['throughput_rps']):.4f} | {float(service['throughput_rps']):.4f} | {deltas['throughput_rps_delta_service_minus_batched']:+.4f} |"
    )
    lines.append(
        f"| p95_latency_ms | {float(batched['p95_latency_ms']):.2f} | {float(service['p95_latency_ms']):.2f} | {deltas['p95_latency_ms_delta_service_minus_batched']:+.2f} |"
    )
    lines.append(
        f"| completion_rate | {float(batched['completion_rate']):.4f} | {float(service['completion_rate']):.4f} | {deltas['completion_rate_delta_service_minus_batched']:+.4f} |"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Batched mode is usually better for throughput-centric bulk processing.")
    lines.append("- Service-style mode is usually better for internal repeated-request loops with lower turnaround needs.")
    lines.append("- Use cloud-native path only when always-on endpoint lifecycle dominates.")

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"REPORT_JSON={report_json}")
    print(f"REPORT_MD={report_md}")
    print(f"RECOMMENDATION={recommendation}")


if __name__ == "__main__":
    main()

