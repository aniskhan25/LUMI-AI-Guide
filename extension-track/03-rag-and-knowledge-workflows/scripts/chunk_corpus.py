#!/usr/bin/env python3
"""Chunk corpus documents into retrievable units with stable IDs."""

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
    parser.add_argument("--corpus-jsonl", type=Path, default=None)
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunk_words(words: List[str], chunk_size: int, overlap: int, min_words: int) -> List[tuple[int, int, str]]:
    if chunk_size <= overlap:
        raise SystemExit("chunk_words must be > overlap_words")
    if not words:
        return []
    step = chunk_size - overlap
    chunks: List[tuple[int, int, str]] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        span = words[start:end]
        if len(span) < min_words and chunks:
            break
        chunks.append((start, end, " ".join(span)))
        if end == len(words):
            break
        start += step
    return chunks


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = run_dir_from(cfg, args.output_root, args.run_name)

    corpus_path = args.corpus_jsonl or Path(str(cfg["data"]["corpus_jsonl"]))
    chunks_path = run_dir / str(cfg["output"]["chunks_jsonl"])

    doc_id_key = str(cfg["data"]["doc_id_key"])
    title_key = str(cfg["data"]["title_key"])
    text_key = str(cfg["data"]["text_key"])
    metadata_key = str(cfg["data"]["metadata_key"])

    chunk_size = int(cfg["chunking"]["chunk_words"])
    overlap = int(cfg["chunking"]["overlap_words"])
    min_words = int(cfg["chunking"]["min_chunk_words"])

    docs = read_jsonl(corpus_path)
    if not docs:
        raise SystemExit(f"No corpus records found in {corpus_path}")

    total_chunks = 0
    with chunks_path.open("w", encoding="utf-8") as out_f:
        for doc in docs:
            doc_id = str(doc[doc_id_key])
            title = str(doc.get(title_key, ""))
            text = str(doc[text_key]).strip()
            metadata = doc.get(metadata_key, {})
            words = text.split()
            spans = chunk_words(words, chunk_size, overlap, min_words)
            if not spans:
                continue
            for idx, (start, end, chunk_text) in enumerate(spans):
                chunk_id = f"{doc_id}-c{idx:04d}"
                out = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_index": idx,
                    "start_word": start,
                    "end_word": end,
                    "chunk_text": chunk_text,
                    "metadata": metadata,
                }
                out_f.write(json.dumps(out) + "\n")
                total_chunks += 1

    print(f"DOC_COUNT={len(docs)}")
    print(f"CHUNK_COUNT={total_chunks}")
    print(f"CHUNKS_PATH={chunks_path}")


if __name__ == "__main__":
    main()
