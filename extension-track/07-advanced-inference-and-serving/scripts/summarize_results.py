#!/usr/bin/env python3
"""Compare batched and service-style summaries and produce decision report."""

from pathlib import Path

from _common import load_yaml, read_json, resolve_path, write_json


def choose_recommendation(batched, service):
    if float(batched.get("completion_rate", 0.0)) < 1.0 <= float(service.get("completion_rate", 0.0)):
        return "service"
    if float(service.get("completion_rate", 0.0)) < 1.0 <= float(batched.get("completion_rate", 0.0)):
        return "batched"

    if float(batched.get("throughput_rps", 0.0)) >= float(service.get("throughput_rps", 0.0)) and float(
        batched.get("p95_latency_ms", 0.0)
    ) <= float(service.get("p95_latency_ms", 0.0)):
        return "batched"

    if float(service.get("p95_latency_ms", 0.0)) < float(batched.get("p95_latency_ms", 0.0)) and float(
        service.get("completion_rate", 0.0)
    ) >= float(batched.get("completion_rate", 0.0)):
        return "service"

    return "inconclusive"


def interpretation_lines(batched, service, recommendation):
    lines = []
    if float(batched["throughput_rps"]) > float(service["throughput_rps"]):
        lines.append("- Batched mode delivered higher throughput on this request set.")
    elif float(service["throughput_rps"]) > float(batched["throughput_rps"]):
        lines.append("- Service-style mode delivered higher throughput on this request set.")

    if float(batched["p95_latency_ms"]) < float(service["p95_latency_ms"]):
        lines.append("- Batched mode had lower p95 latency in this controlled run.")
    elif float(service["p95_latency_ms"]) < float(batched["p95_latency_ms"]):
        lines.append("- Service-style mode had lower p95 latency in this controlled run.")

    if recommendation == "batched":
        lines.append("- Batched mode is the better fit when queued throughput is the dominant requirement.")
    elif recommendation == "service":
        lines.append("- Service-style mode is the better fit when lower turnaround inside one allocation matters more.")
    else:
        lines.append("- The result is inconclusive; tune batch size or concurrency and rerun the same request set.")

    return lines


def main():
    lesson_dir = Path(__file__).resolve().parents[1]
    configs_dir = lesson_dir / "configs"
    inference_cfg = load_yaml(configs_dir / "inference.yaml")
    service_cfg = load_yaml(configs_dir / "service.yaml")

    outputs_dir = resolve_path(configs_dir, str(inference_cfg["run"]["output_dir"]))
    batched_summary_path = outputs_dir / str(inference_cfg["run"]["run_name"]) / str(inference_cfg["output"]["summary_json"])
    service_summary_path = outputs_dir / str(service_cfg["run"]["run_name"]) / str(service_cfg["output"]["summary_json"])

    batched = read_json(batched_summary_path)
    service = read_json(service_summary_path)
    recommendation = choose_recommendation(batched, service)

    deltas = {
        "throughput_rps_delta_service_minus_batched": float(service["throughput_rps"]) - float(batched["throughput_rps"]),
        "p95_latency_ms_delta_service_minus_batched": float(service["p95_latency_ms"]) - float(batched["p95_latency_ms"]),
        "completion_rate_delta_service_minus_batched": float(service["completion_rate"]) - float(batched["completion_rate"]),
        "error_rate_delta_service_minus_batched": float(service["error_rate"]) - float(batched["error_rate"]),
    }

    payload = {
        "batched_summary_path": str(batched_summary_path),
        "service_summary_path": str(service_summary_path),
        "deltas": deltas,
        "recommendation": recommendation,
    }

    report_json = outputs_dir / "advanced-inference-comparison.json"
    report_md = outputs_dir / "advanced-inference-comparison.md"
    write_json(report_json, payload)

    lines = [
        "# Advanced Inference Comparison Report",
        "",
        f"- Recommendation: `{recommendation}`",
        "",
        "| Metric | Batched | Service | Delta (service-batched) |",
        "|---|---:|---:|---:|",
        f"| throughput_rps | {float(batched['throughput_rps']):.4f} | {float(service['throughput_rps']):.4f} | {deltas['throughput_rps_delta_service_minus_batched']:+.4f} |",
        f"| p95_latency_ms | {float(batched['p95_latency_ms']):.2f} | {float(service['p95_latency_ms']):.2f} | {deltas['p95_latency_ms_delta_service_minus_batched']:+.2f} |",
        f"| completion_rate | {float(batched['completion_rate']):.4f} | {float(service['completion_rate']):.4f} | {deltas['completion_rate_delta_service_minus_batched']:+.4f} |",
        f"| error_rate | {float(batched['error_rate']):.4f} | {float(service['error_rate']):.4f} | {deltas['error_rate_delta_service_minus_batched']:+.4f} |",
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(interpretation_lines(batched, service, recommendation))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Batched mode is for queued bulk work where throughput is the main objective.",
            "- Service-style mode is for repeated internal requests inside one scheduled allocation, not a public always-on endpoint.",
            "- If throughput and latency disagree, treat the result as an operating-pattern tradeoff, not a single winner.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"REPORT_JSON={report_json}")
    print(f"REPORT_MD={report_md}")
    print(f"RECOMMENDATION={recommendation}")


if __name__ == "__main__":
    main()
