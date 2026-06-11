#!/usr/bin/env python3
"""Score one variant against the evaluation set and references."""

import argparse
import json
from pathlib import Path

from _common import load_config, read_jsonl, resolve_path, resolve_run_dir, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", type=str, default="baseline")
    return parser.parse_args()


def term_score(answer, required_terms):
    if not required_terms:
        return 0.0
    answer_l = answer.lower()
    hits = sum(1 for term in required_terms if term.lower() in answer_l)
    return hits / len(required_terms)


def choose_failure_category(completion, retrieval_hit, answer_score, grounded, min_score):
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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_root = resolve_run_dir(cfg, args.config)

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
    out_rows = read_jsonl(outputs_path)

    ref_by_id = {str(row["query_id"]): row for row in ref_rows}
    out_by_id = {str(row["query_id"]): row for row in out_rows}

    min_score = float(cfg["metrics"]["min_answer_score"])
    scored = []

    for item in eval_rows:
        query_id = str(item["query_id"])
        out = out_by_id.get(query_id, {})
        ref = ref_by_id.get(query_id, {})
        answer = str(out.get("answer", "")).strip()
        evidence_ids = [str(x) for x in out.get("evidence_chunk_ids", [])]
        expected_doc_id = str(item.get("expected_doc_id", ""))
        retrieval_hit = int(any(chunk_id.startswith(expected_doc_id) for chunk_id in evidence_ids)) if expected_doc_id else 0
        completion = int(bool(answer))
        required_terms = [str(x) for x in ref.get("required_terms", [])]
        answer_score = term_score(answer, required_terms) if completion else 0.0
        grounded = int(retrieval_hit == 1 and answer_score > 0.0 and len(evidence_ids) > 0)
        failure_category = choose_failure_category(completion, retrieval_hit, answer_score, grounded, min_score)

        scored.append(
            {
                "query_id": query_id,
                "variant": variant_name,
                "category": item.get("category", ""),
                "retrieval_hit": retrieval_hit,
                "answer_score": answer_score,
                "grounded": grounded,
                "completion": completion,
                "failure_category": failure_category,
            }
        )

    item_count = len(scored)
    retrieval_hit_rate = sum(row["retrieval_hit"] for row in scored) / max(1, item_count)
    answer_score_mean = sum(row["answer_score"] for row in scored) / max(1, item_count)
    grounded_rate = sum(row["grounded"] for row in scored) / max(1, item_count)
    completion_rate = sum(row["completion"] for row in scored) / max(1, item_count)
    pass_rate = sum(1 for row in scored if row["failure_category"] == "pass") / max(1, item_count)

    scored_path = variant_dir / "scored_records.jsonl"
    write_jsonl(scored_path, scored)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    gpu_visible_count = metadata.get("rag_summary", {}).get("gpu_visible_count")

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
