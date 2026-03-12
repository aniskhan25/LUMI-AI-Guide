#!/usr/bin/env python3
"""Score system outputs against evaluation set and reference answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _common import load_config, read_jsonl, resolve_path, resolve_run_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def term_score(answer: str, required_terms: List[str]) -> float:
    if not required_terms:
        return 0.0
    answer_l = answer.lower()
    hits = sum(1 for t in required_terms if t.lower() in answer_l)
    return hits / len(required_terms)


def choose_failure_category(completion: int, retrieval_hit: int, answer_score: float, grounded: int, min_score: float) -> str:
    if completion == 0:
        return "output_missing_or_empty"
    if retrieval_hit == 0:
        return "retrieved_wrong_document"
    if grounded == 0:
        return "answer_unsupported_by_evidence"
    if answer_score < min_score:
        return "correct_evidence_answer_wrong"
    if answer_score < 1.0:
        return "answer_incomplete"
    return "pass"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = resolve_run_dir(cfg, args.config, args.output_root, args.run_name)

    if args.variant not in cfg["variants"]:
        raise SystemExit(f"Unknown variant: {args.variant}")
    variant_name = str(cfg["variants"][args.variant]["name"])
    variant_dir = run_root / variant_name

    eval_set_path = resolve_path(args.config.parent, str(cfg["paths"]["eval_set_jsonl"]))
    ref_path = resolve_path(args.config.parent, str(cfg["paths"]["reference_answers_jsonl"]))
    outputs_path = variant_dir / "system_outputs.jsonl"
    metadata_path = variant_dir / "run_metadata.json"

    eval_rows = read_jsonl(eval_set_path)
    ref_rows = read_jsonl(ref_path)
    out_rows = read_jsonl(outputs_path) if outputs_path.is_file() else []

    ref_by_id = {str(x["query_id"]): x for x in ref_rows}
    out_by_id = {str(x["query_id"]): x for x in out_rows}

    min_score = float(cfg["metrics"]["min_answer_score"])
    scored: List[Dict[str, Any]] = []

    for item in eval_rows:
        qid = str(item["query_id"])
        out = out_by_id.get(qid, {})
        ref = ref_by_id.get(qid, {})
        answer = str(out.get("answer", "")).strip()
        evidence_ids = [str(x) for x in out.get("evidence_chunk_ids", [])]
        expected_doc_id = str(item.get("expected_doc_id", ""))
        retrieval_hit = int(any(cid.startswith(expected_doc_id) for cid in evidence_ids)) if expected_doc_id else 0
        completion = int(bool(answer))
        required_terms = [str(x) for x in ref.get("required_terms", [])]
        a_score = term_score(answer, required_terms) if completion else 0.0
        grounded = int(retrieval_hit == 1 and a_score > 0.0 and len(evidence_ids) > 0)
        failure = choose_failure_category(completion, retrieval_hit, a_score, grounded, min_score)

        scored.append(
            {
                "query_id": qid,
                "variant": variant_name,
                "category": item.get("category", ""),
                "retrieval_hit": retrieval_hit,
                "answer_score": a_score,
                "grounded": grounded,
                "completion": completion,
                "failure_category": failure,
            }
        )

    item_count = len(scored)
    retrieval_hit_rate = sum(x["retrieval_hit"] for x in scored) / max(1, item_count)
    answer_score_mean = sum(x["answer_score"] for x in scored) / max(1, item_count)
    grounded_rate = sum(x["grounded"] for x in scored) / max(1, item_count)
    completion_rate = sum(x["completion"] for x in scored) / max(1, item_count)
    pass_rate = sum(1 for x in scored if x["failure_category"] == "pass") / max(1, item_count)

    scored_path = variant_dir / "scored_records.jsonl"
    write_jsonl(scored_path, scored)

    gpu_visible_count = None
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        gpu_visible_count = m.get("rag_summary", {}).get("gpu_visible_count")

    summary = {
        "variant": variant_name,
        "item_count": item_count,
        "retrieval_hit_rate": retrieval_hit_rate,
        "answer_score_mean": answer_score_mean,
        "grounded_rate": grounded_rate,
        "completion_rate": completion_rate,
        "pass_rate": pass_rate,
        "gpu_visible_count": gpu_visible_count,
    }
    summary_path = variant_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"VARIANT={variant_name}")
    print(f"SCORED_RECORDS={scored_path}")
    print(f"SUMMARY={summary_path}")
    print(f"ITEM_COUNT={item_count}")


if __name__ == "__main__":
    main()

