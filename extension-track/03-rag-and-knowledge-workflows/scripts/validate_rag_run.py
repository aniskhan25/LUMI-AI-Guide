#!/usr/bin/env python3
"""Validate Lesson 03 RAG artifacts."""

import argparse
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dir_from(cfg):
    return Path(str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_file(path):
    if not path.is_file():
        raise SystemExit(f"Missing expected file: {path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = run_dir_from(cfg)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    corpus_path = Path(str(cfg["data"]["corpus_jsonl"]))
    queries_path = Path(str(cfg["data"]["queries_jsonl"]))
    chunks_path = run_dir / str(cfg["output"]["chunks_jsonl"])
    embeddings_path = run_dir / str(cfg["output"]["embeddings_jsonl"])
    retrieval_path = run_dir / str(cfg["output"]["retrieval_results_jsonl"])
    answers_path = run_dir / str(cfg["output"]["answers_jsonl"])
    summary_path = run_dir / str(cfg["output"]["summary_json"])

    for path in [corpus_path, queries_path, chunks_path, embeddings_path, retrieval_path, answers_path, summary_path]:
        ensure_file(path)

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

    chunk_ids = [str(row["chunk_id"]) for row in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise SystemExit("Duplicate chunk_id values found in chunks.jsonl")

    embedding_ids = [str(row["chunk_id"]) for row in embeddings]
    if set(embedding_ids) != set(chunk_ids):
        raise SystemExit("Mismatch between chunk IDs and embedding IDs")

    first_embedding = embeddings[0].get("embedding")
    if not isinstance(first_embedding, list) or not first_embedding:
        raise SystemExit("First embedding vector is missing or empty")
    embedding_dim = len(first_embedding)
    for idx, row in enumerate(embeddings):
        vector = row.get("embedding")
        if not isinstance(vector, list) or len(vector) != embedding_dim:
            raise SystemExit(f"Inconsistent embedding dimension at row {idx}")

    query_ids = {str(row["query_id"]) for row in queries}
    retrieval_ids = {str(row["query_id"]) for row in retrieval}
    answer_ids = {str(row["query_id"]) for row in answers}
    if query_ids != retrieval_ids:
        raise SystemExit("Mismatch between query IDs and retrieval results")
    if query_ids != answer_ids:
        raise SystemExit("Mismatch between query IDs and answers")

    chunk_id_set = set(chunk_ids)
    for row in retrieval:
        retrieved = row.get("retrieved", [])
        if not isinstance(retrieved, list) or not retrieved:
            raise SystemExit(f"Empty retrieved set for query_id={row.get('query_id')}")
        for item in retrieved:
            chunk_id = str(item.get("chunk_id", ""))
            if chunk_id not in chunk_id_set:
                raise SystemExit(f"Retrieved chunk_id not found in chunk manifest: {chunk_id}")

    for row in answers:
        answer = str(row.get("answer", "")).strip()
        if not answer:
            raise SystemExit(f"Empty answer text for query_id={row.get('query_id')}")
        evidence = row.get("evidence_chunk_ids", [])
        if not isinstance(evidence, list) or not evidence:
            raise SystemExit(f"Missing evidence_chunk_ids for query_id={row.get('query_id')}")
        for chunk_id in evidence:
            if str(chunk_id) not in chunk_id_set:
                raise SystemExit(f"Answer references missing chunk_id: {chunk_id}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if int(summary.get("gpu_visible_count", 0)) < 1:
        raise SystemExit("Summary reports gpu_visible_count < 1")
    if int(summary.get("query_count", -1)) != len(queries):
        raise SystemExit("Summary query_count does not match queries file")
    if int(summary.get("chunk_count", -1)) != len(chunks):
        raise SystemExit("Summary chunk_count does not match chunks file")

    print("VALIDATION_OK=1")
    print(f"docs={len(corpus)}")
    print(f"chunks={len(chunks)}")
    print(f"embedding_dim={embedding_dim}")
    print(f"queries={len(queries)}")
    print(f"answers={len(answers)}")


if __name__ == "__main__":
    main()
