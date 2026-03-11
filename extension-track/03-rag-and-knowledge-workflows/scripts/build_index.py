#!/usr/bin/env python3
"""Build a simple retriever index from chunk embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml is required. Install PyYAML or run inside the AI Factory container.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dir_from(cfg: Dict[str, Any], output_root: Path | None, run_name: str | None) -> Path:
    name = run_name or str(cfg["run"]["run_name"])
    root = output_root or Path(str(cfg["run"]["output_dir"]))
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_embeddings(path: Path) -> Tuple[List[str], np.ndarray]:
    ids: List[str] = []
    vectors: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(str(row["chunk_id"]))
            vectors.append([float(x) for x in row["embedding"]])
    if not ids:
        raise SystemExit(f"No embeddings found in {path}")
    matrix = np.asarray(vectors, dtype=np.float32)
    return ids, matrix


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = run_dir_from(cfg, args.output_root, args.run_name)

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
