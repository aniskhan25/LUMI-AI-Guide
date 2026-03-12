#!/usr/bin/env python3
"""Collect latency/throughput metrics for one run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from _common import (
    load_yaml,
    percentile,
    read_json,
    read_jsonl,
    resolve_run_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["batched", "service"], required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config, args.output_root, args.run_name)

    requests_path = run_dir / str(cfg["output"]["requests_copy_jsonl"])
    responses_path = run_dir / str(cfg["output"]["responses_jsonl"])
    errors_path = run_dir / str(cfg["output"]["errors_jsonl"])
    metadata_path = run_dir / str(cfg["output"]["run_metadata_json"])
    metrics_path = run_dir / str(cfg["output"]["metrics_json"])
    summary_path = run_dir / str(cfg["output"]["summary_json"])

    requests = read_jsonl(requests_path)
    responses = read_jsonl(responses_path) if responses_path.is_file() else []
    errors = read_jsonl(errors_path) if errors_path.is_file() else []
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}

    latencies: List[float] = [float(r.get("latency_ms", 0.0)) for r in responses]
    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    p99 = percentile(latencies, 0.99)
    total_latency_ms = sum(latencies)

    if responses:
        first_start = min(int(r.get("start_ts", 0)) for r in responses)
        last_end = max(int(r.get("end_ts", 0)) for r in responses)
        duration_s = max(1e-9, (last_end - first_start) / 1000.0)
    else:
        duration_s = 1e-9

    processed = len(responses) + len(errors)
    throughput_rps = len(responses) / duration_s
    completion_rate = len(responses) / max(1, len(requests))
    error_rate = len(errors) / max(1, len(requests))

    metrics = {
        "run_name": run_dir.name,
        "mode": args.mode,
        "processed_count": processed,
        "response_count": len(responses),
        "error_count": len(errors),
        "duration_seconds": duration_s,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "total_latency_ms": total_latency_ms,
        "throughput_rps": throughput_rps,
        "completion_rate": completion_rate,
        "error_rate": error_rate,
    }
    write_json(metrics_path, metrics)

    summary = {
        **metrics,
        "model_id": metadata.get("model_id", ""),
        "gpu_visible_count": metadata.get("gpu_visible_count", 0),
        "batch_size": metadata.get("batch_size", 0),
        "concurrency": metadata.get("concurrency", 0),
        "request_count": len(requests),
    }
    write_json(summary_path, summary)

    print(f"SUMMARY_PATH={summary_path}")
    print(f"THROUGHPUT_RPS={throughput_rps:.4f}")
    print(f"P95_MS={p95:.2f}")


if __name__ == "__main__":
    main()

