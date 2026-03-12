#!/usr/bin/env python3
"""Aggregate per-rank raw metrics into a run summary."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List

from _common import list_json_files, load_yaml, read_json, resolve_run_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config, args.output_root, args.run_name)
    raw_dir = run_dir / str(cfg["output"]["raw_dir"])

    metric_files = list_json_files(raw_dir, str(cfg["output"]["metrics_prefix"]))
    placement_files = list_json_files(raw_dir, str(cfg["output"]["placement_prefix"]))
    if not metric_files:
        raise SystemExit(f"No metrics files found in {raw_dir}")

    metrics: List[Dict[str, Any]] = [read_json(p) for p in metric_files]
    placements: List[Dict[str, Any]] = [read_json(p) for p in placement_files] if placement_files else []

    world_sizes = {int(m["world_size"]) for m in metrics}
    if len(world_sizes) != 1:
        raise SystemExit(f"Inconsistent world sizes in metrics files: {world_sizes}")
    world_size = world_sizes.pop()

    expected_world_size = int(cfg["distributed"]["expected_world_size"])
    expected_nodes = int(cfg["distributed"]["expected_nodes"])
    hostnames = sorted({p.get("hostname", "") for p in placements if p.get("hostname", "")})
    node_count = len(hostnames) if hostnames else int(metrics[0].get("node_count", 0) or 0)

    throughputs = [float(m["throughput_samples_per_sec"]) for m in metrics]
    elapsed = [float(m["elapsed_seconds"]) for m in metrics]
    samples_per_step = int(metrics[0]["samples_per_step"])
    gpu_visible_count = int(metrics[0].get("gpu_visible_count", 0))

    summary = {
        "run_name": run_dir.name,
        "world_size": world_size,
        "expected_world_size": expected_world_size,
        "world_size_matches_expected": world_size == expected_world_size,
        "node_count": node_count,
        "expected_nodes": expected_nodes,
        "node_count_matches_expected": (node_count == expected_nodes) if node_count else None,
        "gpu_visible_count": gpu_visible_count,
        "rank_count": len(metrics),
        "effective_samples_per_step": samples_per_step * world_size,
        "mean_rank_throughput_samples_per_sec": statistics.mean(throughputs),
        "total_throughput_samples_per_sec": sum(throughputs),
        "max_elapsed_seconds": max(elapsed),
        "min_elapsed_seconds": min(elapsed),
        "hostnames": hostnames,
        "raw_metrics_files": [str(p) for p in metric_files],
        "raw_placement_files": [str(p) for p in placement_files],
    }

    out_path = run_dir / str(cfg["output"]["run_summary_json"])
    write_json(out_path, summary)

    print(f"RUN_SUMMARY={out_path}")
    print(f"WORLD_SIZE={world_size}")
    print(f"TOTAL_THROUGHPUT={summary['total_throughput_samples_per_sec']:.4f}")


if __name__ == "__main__":
    main()

