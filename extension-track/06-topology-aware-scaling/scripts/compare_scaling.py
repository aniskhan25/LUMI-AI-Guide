#!/usr/bin/env python3
"""Compare scaling summaries and build a compact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _common import load_yaml, read_json, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-config", type=Path, required=True)
    return parser.parse_args()


def speedup(base: float, target: float) -> float:
    return target / max(1e-9, base)


def efficiency(spd: float, base_world: int, target_world: int) -> float:
    ratio = target_world / max(1, base_world)
    return spd / max(1e-9, ratio)


def diagnose(eff: float) -> str:
    if eff >= 0.8:
        return "good scaling efficiency"
    if eff >= 0.5:
        return "moderate scaling efficiency; inspect communication and placement"
    return "poor scaling efficiency; likely communication or mapping bottleneck"


def metric_row(label: str, summary: Dict[str, Any], base_thr: float, base_world: int) -> Dict[str, Any]:
    thr = float(summary["total_throughput_samples_per_sec"])
    ws = int(summary["world_size"])
    spd = speedup(base_thr, thr)
    eff = efficiency(spd, base_world, ws)
    return {
        "label": label,
        "world_size": ws,
        "node_count": int(summary.get("node_count", 0) or 0),
        "total_throughput_samples_per_sec": thr,
        "speedup_vs_baseline": spd,
        "efficiency_vs_baseline": eff,
        "diagnosis": diagnose(eff),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.compare_config)
    base_dir = args.compare_config.parent

    baseline_summary = read_json(resolve_path(base_dir, str(cfg["paths"]["baseline_summary"])))
    single_summary = read_json(resolve_path(base_dir, str(cfg["paths"]["single_node_summary"])))
    multi_summary = read_json(resolve_path(base_dir, str(cfg["paths"]["multi_node_summary"])))

    base_thr = float(baseline_summary["total_throughput_samples_per_sec"])
    base_world = int(baseline_summary["world_size"])

    rows: List[Dict[str, Any]] = []
    rows.append(metric_row(str(cfg["comparison"]["baseline_label"]), baseline_summary, base_thr, base_world))
    rows.append(metric_row(str(cfg["comparison"]["single_node_label"]), single_summary, base_thr, base_world))
    rows.append(metric_row(str(cfg["comparison"]["multi_node_label"]), multi_summary, base_thr, base_world))

    out_json = resolve_path(base_dir, str(cfg["comparison"]["report_json"]))
    out_md = resolve_path(base_dir, str(cfg["comparison"]["report_md"]))

    payload = {
        "baseline_summary_path": str(resolve_path(base_dir, str(cfg["paths"]["baseline_summary"]))),
        "single_node_summary_path": str(resolve_path(base_dir, str(cfg["paths"]["single_node_summary"]))),
        "multi_node_summary_path": str(resolve_path(base_dir, str(cfg["paths"]["multi_node_summary"]))),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines: List[str] = []
    lines.append("# Scaling Comparison Report")
    lines.append("")
    lines.append("| Configuration | World Size | Nodes | Throughput | Speedup | Efficiency | Diagnosis |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['world_size']} | {row['node_count']} | "
            f"{row['total_throughput_samples_per_sec']:.2f} | {row['speedup_vs_baseline']:.2f} | "
            f"{row['efficiency_vs_baseline']:.2f} | {row['diagnosis']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Keep effective workload assumptions consistent across runs.")
    lines.append("- Validate placement metadata before interpreting poor scaling.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"SCALING_REPORT_JSON={out_json}")
    print(f"SCALING_REPORT_MD={out_md}")


if __name__ == "__main__":
    main()

