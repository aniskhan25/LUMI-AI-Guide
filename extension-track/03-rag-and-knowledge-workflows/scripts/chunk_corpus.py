#!/usr/bin/env python3
"""Chunk corpus documents into retrievable units with stable IDs."""

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
    run_dir = Path(str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunk_words(words, chunk_size, overlap, min_words):
    if chunk_size <= overlap:
        raise SystemExit("chunk_words must be greater than overlap_words")
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = run_dir_from(cfg)

    corpus_path = Path(str(cfg["data"]["corpus_jsonl"]))
    chunks_path = run_dir / str(cfg["output"]["chunks_jsonl"])

    docs = read_jsonl(corpus_path)
    if not docs:
        raise SystemExit(f"No corpus records found in {corpus_path}")

    doc_id_key = str(cfg["data"]["doc_id_key"])
    title_key = str(cfg["data"]["title_key"])
    text_key = str(cfg["data"]["text_key"])
    metadata_key = str(cfg["data"]["metadata_key"])

    chunk_size = int(cfg["chunking"]["chunk_words"])
    overlap = int(cfg["chunking"]["overlap_words"])
    min_words = int(cfg["chunking"]["min_chunk_words"])

    chunk_count = 0
    with chunks_path.open("w", encoding="utf-8") as out_f:
        for doc in docs:
            doc_id = str(doc[doc_id_key])
            title = str(doc.get(title_key, ""))
            metadata = doc.get(metadata_key, {})
            spans = chunk_words(str(doc[text_key]).split(), chunk_size, overlap, min_words)
            for idx, (start, end, chunk_text) in enumerate(spans):
                row = {
                    "chunk_id": f"{doc_id}-c{idx:04d}",
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_index": idx,
                    "start_word": start,
                    "end_word": end,
                    "chunk_text": chunk_text,
                    "metadata": metadata,
                }
                out_f.write(json.dumps(row) + "\n")
                chunk_count += 1

    print(f"DOC_COUNT={len(docs)}")
    print(f"CHUNK_COUNT={chunk_count}")
    print(f"CHUNKS_PATH={chunks_path}")


if __name__ == "__main__":
    main()
