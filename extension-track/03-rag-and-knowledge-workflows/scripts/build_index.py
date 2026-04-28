#!/usr/bin/env python3
"""Build a simple local retriever index."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dir_from(cfg):
    run_dir = Path(str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_embeddings(path):
    chunk_ids = []
    vectors = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunk_ids.append(str(row["chunk_id"]))
            vectors.append([float(x) for x in row["embedding"]])
    if not chunk_ids:
        raise SystemExit(f"No embeddings found in {path}")
    return chunk_ids, np.asarray(vectors, dtype=np.float32)


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = run_dir_from(cfg)

    embeddings_path = run_dir / str(cfg["output"]["embeddings_jsonl"])
    index_path = run_dir / str(cfg["output"]["retriever_index_npz"])

    chunk_ids, matrix = load_embeddings(embeddings_path)
    if str(cfg["retrieval"]["score_type"]).lower() == "cosine":
        matrix = normalize_rows(matrix)

    np.savez(index_path, chunk_ids=np.asarray(chunk_ids), embeddings=matrix)

    print(f"INDEX_PATH={index_path}")
    print(f"INDEX_ROWS={matrix.shape[0]}")
    print(f"INDEX_DIM={matrix.shape[1]}")


if __name__ == "__main__":
    main()
