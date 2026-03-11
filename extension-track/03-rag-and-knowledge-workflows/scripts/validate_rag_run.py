#!/usr/bin/env python3
"""Validate Lesson 03 RAG artifacts and cross-file consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml is required. Install PyYAML or run inside the AI Factory container.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dir_from(cfg: Dict[str, Any], output_root: Path | None, run_name: str | None) -> Path:
    name = run_name or str(cfg["run"]["run_name"])
    root = output_root or Path(str(cfg["run"]["output_dir"]))
    return root / name


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing expected file: {path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = run_dir_from(cfg, args.output_root, args.run_name)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    corpus_path = Path(str(cfg["data"]["corpus_jsonl"]))
    queries_path = Path(str(cfg["data"]["queries_jsonl"]))
    chunks_path = run_dir / str(cfg["output"]["chunks_jsonl"])
    embeddings_path = run_dir / str(cfg["output"]["embeddings_jsonl"])
    retrieval_path = run_dir / str(cfg["output"]["retrieval_results_jsonl"])
    answers_path = run_dir / str(cfg["output"]["answers_jsonl"])
    summary_path = run_dir / str(cfg["output"]["summary_json"])

    for p in [corpus_path, queries_path, chunks_path, embeddings_path, retrieval_path, answers_path, summary_path]:
        ensure_file(p)

    corpus = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)
    chunks = read_jsonl(chunks_path)
    embeddings = read_jsonl(embeddings_path)
    retrieval = read_jsonl(retrieval_path)
    answers = read_jsonl(answers_path)

    if not corpus:
        raise SystemExit("Corpus is empty")
    if not queries:
        raise SystemExit("Queries are empty")
    if not chunks:
        raise SystemExit("Chunks are empty")
    if not embeddings:
        raise SystemExit("Embeddings are empty")
    if not retrieval:
        raise SystemExit("Retrieval results are empty")
    if not answers:
        raise SystemExit("Answers are empty")

    doc_ids = {str(x["doc_id"]) for x in corpus}
    chunk_doc_ids = {str(x["doc_id"]) for x in chunks}
    if not doc_ids.issubset(chunk_doc_ids):
        missing = sorted(doc_ids - chunk_doc_ids)
        raise SystemExit(f"Some corpus docs have no chunks: {missing[:5]}")

    chunk_ids = [str(x["chunk_id"]) for x in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise SystemExit("Duplicate chunk_id values found in chunk manifest")

    emb_chunk_ids = [str(x["chunk_id"]) for x in embeddings]
    if len(emb_chunk_ids) != len(set(emb_chunk_ids)):
        raise SystemExit("Duplicate chunk_id values found in embeddings file")
    if set(emb_chunk_ids) != set(chunk_ids):
        raise SystemExit("Mismatch between chunk IDs and embedding IDs")

    first_vec = embeddings[0].get("embedding")
    if not isinstance(first_vec, list) or len(first_vec) == 0:
        raise SystemExit("First embedding vector is missing or empty")
    dim = len(first_vec)
    for i, row in enumerate(embeddings):
        vec = row.get("embedding")
        if not isinstance(vec, list) or len(vec) != dim:
            raise SystemExit(f"Inconsistent embedding dimension at row {i}")

    query_ids = {str(x["query_id"]) for x in queries}
    retrieval_query_ids = {str(x["query_id"]) for x in retrieval}
    answer_query_ids = {str(x["query_id"]) for x in answers}
    if query_ids != retrieval_query_ids:
        raise SystemExit("Mismatch between query IDs and retrieval result IDs")
    if query_ids != answer_query_ids:
        raise SystemExit("Mismatch between query IDs and answer IDs")

    chunk_id_set = set(chunk_ids)
    for row in retrieval:
        retrieved = row.get("retrieved", [])
        if not isinstance(retrieved, list) or len(retrieved) == 0:
            raise SystemExit(f"Empty retrieved set for query_id={row.get('query_id')}")
        for item in retrieved:
            cid = str(item.get("chunk_id", ""))
            if cid not in chunk_id_set:
                raise SystemExit(f"Retrieved chunk_id not in chunk manifest: {cid}")

    for row in answers:
        answer = str(row.get("answer", "")).strip()
        if not answer:
            raise SystemExit(f"Empty answer text for query_id={row.get('query_id')}")
        evidence = row.get("evidence_chunk_ids", [])
        if not isinstance(evidence, list) or len(evidence) == 0:
            raise SystemExit(f"Missing evidence_chunk_ids for query_id={row.get('query_id')}")
        for cid in evidence:
            if str(cid) not in chunk_id_set:
                raise SystemExit(f"Answer references missing chunk_id: {cid}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if int(summary.get("query_count", -1)) != len(queries):
        raise SystemExit("Summary query_count does not match queries file")
    if int(summary.get("chunk_count", -1)) != len(chunks):
        raise SystemExit("Summary chunk_count does not match chunks file")
    if args.require_gpu and int(summary.get("gpu_visible_count", 0)) < 1:
        raise SystemExit("GPU required but summary reports gpu_visible_count < 1")

    print("VALIDATION_OK=1")
    print(f"docs={len(corpus)}")
    print(f"chunks={len(chunks)}")
    print(f"embedding_dim={dim}")
    print(f"queries={len(queries)}")
    print(f"answers={len(answers)}")


if __name__ == "__main__":
    main()
