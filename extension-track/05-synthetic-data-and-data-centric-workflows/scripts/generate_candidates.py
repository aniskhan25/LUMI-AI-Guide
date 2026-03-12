#!/usr/bin/env python3
"""Generate synthetic candidates from selected weak cases."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from _common import load_yaml, read_jsonl, resolve_path, resolve_run_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def detect_gpu(require_gpu: bool, allow_cpu_fallback: bool) -> int:
    try:
        import torch
    except ImportError:
        if require_gpu:
            raise SystemExit("torch is required to enforce GPU visibility.")
        return 0

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if require_gpu and visible < 1 and not allow_cpu_fallback:
        raise SystemExit("GPU required for generation but no CUDA device is visible.")
    return visible


def make_question(base: str, style: str) -> str:
    if style == "concise":
        return f"Provide a direct answer: {base}"
    if style == "procedural":
        return f"Explain step-by-step for operations teams: {base}"
    if style == "checklist":
        return f"Answer in checklist form with required actions: {base}"
    return base


def make_answer(reference: str, style: str, rnd: random.Random) -> str:
    lead = {
        "concise": "Key answer:",
        "procedural": "Procedure:",
        "checklist": "Checklist:",
    }.get(style, "Answer:")
    suffixes = [
        "Preserve exact timing and role details.",
        "Keep evidence-grounded wording.",
        "Include concrete thresholds and actors.",
    ]
    return f"{lead} {reference} {rnd.choice(suffixes)}".strip()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.generate_config)
    run_dir = resolve_run_dir(cfg, args.generate_config, args.output_root, args.run_name)

    selected_path = run_dir / str(cfg["output"]["selected_weak_cases_jsonl"])
    if selected_path.is_file():
        weak_cases = read_jsonl(selected_path)
    else:
        weak_cases_path = resolve_path(args.generate_config.parent, str(cfg["paths"]["weak_cases_jsonl"]))
        weak_cases = read_jsonl(weak_cases_path)

    if not weak_cases:
        raise SystemExit("No weak cases found. Run identify_weak_cases.py first.")

    num_per_case = int(cfg["generation"]["num_candidates_per_case"])
    styles = [str(x) for x in cfg["generation"]["style_variants"]]
    seed = int(cfg["run"]["seed"])
    rnd = random.Random(seed)

    require_gpu = bool(cfg["generation"]["require_gpu"])
    allow_cpu = bool(cfg["generation"]["allow_cpu_fallback"])
    gpu_visible_count = detect_gpu(require_gpu, allow_cpu)
    print(f"GPU_VISIBLE_COUNT={gpu_visible_count}")

    candidates: List[Dict[str, Any]] = []
    for case in weak_cases:
        case_id = str(case["case_id"])
        base_question = str(case["input_text"])
        ref_answer = str(case["reference_answer"])
        gap_label = str(case["gap_label"])
        failure_type = str(case.get("failure_type", "unknown"))
        required_terms = [str(x) for x in case.get("required_terms", [])]
        evidence_reference = str(case.get("evidence_reference", ""))

        for idx in range(num_per_case):
            style = styles[idx % len(styles)] if styles else "concise"
            synthetic_id = f"{case_id}-s{idx+1:02d}"
            candidates.append(
                {
                    "synthetic_id": synthetic_id,
                    "source_case_id": case_id,
                    "generated_input": make_question(base_question, style),
                    "generated_target": make_answer(ref_answer, style, rnd),
                    "gap_label": gap_label,
                    "failure_type": failure_type,
                    "required_terms": required_terms,
                    "evidence_reference": evidence_reference,
                    "provenance": {
                        "generator": "template-guided-v1",
                        "style": style,
                        "seed": seed,
                    },
                    "filter_status": "pending",
                }
            )

    out_path = run_dir / str(cfg["output"]["candidates_jsonl"])
    write_jsonl(out_path, candidates)

    summary = {
        "weak_case_count": len(weak_cases),
        "num_candidates_per_case": num_per_case,
        "candidate_count": len(candidates),
        "gpu_visible_count": gpu_visible_count,
        "output_path": str(out_path),
    }
    summary_path = run_dir / str(cfg["output"]["generation_summary_json"])
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"CANDIDATE_COUNT={len(candidates)}")
    print(f"CANDIDATES_PATH={out_path}")
    print(f"SUMMARY_PATH={summary_path}")


if __name__ == "__main__":
    main()

