#!/usr/bin/env python3
"""Run one evaluation variant and produce evaluation-ready outputs."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from _common import dump_yaml, load_config, read_jsonl, resolve_path, resolve_run_dir, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", type=str, default="baseline")
    return parser.parse_args()


def run_cmd(cmd, cwd):
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_root = resolve_run_dir(cfg, args.config)

    if args.variant not in cfg["variants"]:
        raise SystemExit(f"Unknown variant: {args.variant}")

    variant_cfg = cfg["variants"][args.variant]
    variant_name = str(variant_cfg["name"])
    variant_dir = run_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    eval_set_path = resolve_path(args.config.parent, str(cfg["paths"]["eval_set_jsonl"]))
    eval_set = read_jsonl(eval_set_path)
    if not eval_set:
        raise SystemExit(f"Evaluation set is empty: {eval_set_path}")

    rag_dir = resolve_path(args.config.parent, str(cfg["paths"]["rag_lesson_dir"]))
    rag_config_path = resolve_path(args.config.parent, str(cfg["paths"]["rag_config"]))
    rag_cfg = load_config(rag_config_path)
    rag_corpus_path = (rag_dir / "data" / "corpus.jsonl").resolve()

    eval_queries_path = variant_dir / "eval_queries.jsonl"
    query_rows = [{"query_id": row["query_id"], "query": row["question"]} for row in eval_set]
    write_jsonl(eval_queries_path, query_rows)

    rag_cfg_effective = copy.deepcopy(rag_cfg)
    rag_cfg_effective["run"]["output_dir"] = str((variant_dir / "rag_artifacts").resolve())
    rag_cfg_effective["run"]["run_name"] = "rag"
    rag_cfg_effective["data"]["corpus_jsonl"] = str(rag_corpus_path)
    rag_cfg_effective["data"]["queries_jsonl"] = str(eval_queries_path.resolve())
    rag_cfg_effective["retrieval"]["top_k"] = int(variant_cfg["top_k"])
    rag_cfg_effective_path = variant_dir / "rag_config_effective.yaml"
    dump_yaml(rag_cfg_effective_path, rag_cfg_effective)

    py = sys.executable
    run_cmd([py, "data/prepare_corpus.py", "--output", "data"], cwd=rag_dir)
    run_cmd([py, "scripts/chunk_corpus.py", "--config", str(rag_cfg_effective_path)], cwd=rag_dir)
    run_cmd([py, "scripts/embed_chunks.py", "--config", str(rag_cfg_effective_path)], cwd=rag_dir)
    run_cmd([py, "scripts/build_index.py", "--config", str(rag_cfg_effective_path)], cwd=rag_dir)
    run_cmd([py, "scripts/answer_queries.py", "--config", str(rag_cfg_effective_path)], cwd=rag_dir)

    rag_run_dir = Path(str(rag_cfg_effective["run"]["output_dir"])) / str(rag_cfg_effective["run"]["run_name"])
    answers_path = rag_run_dir / str(rag_cfg_effective["output"]["answers_jsonl"])
    retrieval_path = rag_run_dir / str(rag_cfg_effective["output"]["retrieval_results_jsonl"])
    summary_path = rag_run_dir / str(rag_cfg_effective["output"]["summary_json"])

    answers_rows = read_jsonl(answers_path)
    retrieval_rows = read_jsonl(retrieval_path)
    answers_by_id = {str(row["query_id"]): row for row in answers_rows}
    retrieval_by_id = {str(row["query_id"]): row for row in retrieval_rows}

    system_outputs = []
    for item in eval_set:
        query_id = str(item["query_id"])
        answer_row = answers_by_id.get(query_id, {})
        retrieval_row = retrieval_by_id.get(query_id, {})
        retrieved = retrieval_row.get("retrieved", [])
        system_outputs.append(
            {
                "query_id": query_id,
                "question": item["question"],
                "answer": answer_row.get("answer", ""),
                "evidence_chunk_ids": answer_row.get("evidence_chunk_ids", []),
                "retrieved_chunk_ids": [row.get("chunk_id", "") for row in retrieved],
                "variant": variant_name,
                "generation_backend": answer_row.get("generation_backend", ""),
            }
        )

    outputs_path = variant_dir / "system_outputs.jsonl"
    write_jsonl(outputs_path, system_outputs)

    with summary_path.open("r", encoding="utf-8") as f:
        rag_summary = json.load(f)

    metadata = {
        "variant": variant_name,
        "top_k": int(variant_cfg["top_k"]),
        "eval_set_path": str(eval_set_path),
        "outputs_path": str(outputs_path),
        "rag_summary_path": str(summary_path),
        "rag_summary": rag_summary,
    }
    metadata_path = variant_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"VARIANT={variant_name}")
    print(f"SYSTEM_OUTPUTS={outputs_path}")
    print(f"METADATA={metadata_path}")
    print(f"ITEMS={len(system_outputs)}")


if __name__ == "__main__":
    main()
