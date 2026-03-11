#!/usr/bin/env python3
"""Batch embedding pipeline for Lesson 02."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_name = args.run_name or str(cfg["run"]["run_name"])
    out_dir = args.output_dir or (Path(str(cfg["run"]["output_dir"])) / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_jsonl = args.input_jsonl or Path(str(cfg["data"]["input_jsonl"]))
    embeddings_path = out_dir / str(cfg["output"]["embeddings_filename"])
    summary_path = out_dir / str(cfg["output"]["summary_filename"])

    set_seed(int(cfg["run"]["seed"]))

    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"GPU_VISIBLE_COUNT={gpu_visible_count}")

    if not torch.cuda.is_available() and not bool(cfg["runtime"]["allow_cpu_fallback"]):
        raise SystemExit("CUDA device not visible. Set runtime.allow_cpu_fallback=true only for local debugging.")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(cfg['runtime']['device_index'])}")
    else:
        device = torch.device("cpu")

    model_name = str(cfg["model"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=bool(cfg["model"]["trust_remote_code"])
    )
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=bool(cfg["model"]["trust_remote_code"])
    ).to(device)
    model.eval()

    rows = read_jsonl(input_jsonl, int(cfg["data"]["max_samples"]))
    if not rows:
        raise SystemExit(f"No input records found in {input_jsonl}")

    id_key = str(cfg["data"]["id_key"])
    text_key = str(cfg["data"]["text_key"])
    metadata_key = str(cfg["data"]["metadata_key"])
    batch_size = int(cfg["inference"]["batch_size"])
    max_seq_len = int(cfg["inference"]["max_seq_len"])
    normalize = bool(cfg["inference"]["normalize_embeddings"])
    log_every = int(cfg["run"]["log_every_batches"])

    start = time.time()
    total = 0
    embedding_dim = None

    with embeddings_path.open("w", encoding="utf-8") as out_f:
        for idx, batch_rows in enumerate(batched(rows, batch_size), start=1):
            texts = [str(r[text_key]) for r in batch_rows]
            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            with torch.no_grad():
                outputs = model(**tokenized)
                pooled = mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            pooled = pooled.detach().cpu().float()
            if embedding_dim is None:
                embedding_dim = int(pooled.shape[1])

            for row, emb in zip(batch_rows, pooled, strict=True):
                out = {
                    "id": row[id_key],
                    "embedding": emb.tolist(),
                }
                if metadata_key in row:
                    out["metadata"] = row[metadata_key]
                out_f.write(json.dumps(out) + "\n")
                total += 1

            if idx % log_every == 0:
                print(f"BATCH={idx} OUTPUT_RECORDS={total}")

    elapsed = time.time() - start
    summary = {
        "mode": "embeddings",
        "run_name": run_name,
        "model_name": model_name,
        "device": str(device),
        "gpu_visible_count": gpu_visible_count,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(embeddings_path),
        "records_written": total,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "elapsed_seconds": elapsed,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("RUN_COMPLETE=1")
    print(f"OUTPUT_JSONL={embeddings_path}")
    print(f"SUMMARY_JSON={summary_path}")


if __name__ == "__main__":
    main()

