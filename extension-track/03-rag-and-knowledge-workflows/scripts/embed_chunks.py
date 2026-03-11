#!/usr/bin/env python3
"""Generate chunk embeddings for Lesson 03 RAG workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml is required. Install PyYAML or run inside the AI Factory container.") from exc
from transformers import AutoModel, AutoTokenizer


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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
    run_dir = run_dir_from(cfg, args.output_root, args.run_name)

    chunks_path = run_dir / str(cfg["output"]["chunks_jsonl"])
    embeddings_path = run_dir / str(cfg["output"]["embeddings_jsonl"])
    rows = read_jsonl(chunks_path)
    if not rows:
        raise SystemExit(f"No chunks found at {chunks_path}. Run chunk_corpus.py first.")

    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"GPU_VISIBLE_COUNT={gpu_visible_count}")
    if not torch.cuda.is_available() and not bool(cfg["runtime"]["allow_cpu_fallback"]):
        raise SystemExit("CUDA device not visible. Set runtime.allow_cpu_fallback=true only for local debugging.")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(cfg['runtime']['device_index'])}")
    else:
        device = torch.device("cpu")

    model_name = str(cfg["embedding"]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=bool(cfg["embedding"]["trust_remote_code"])
    )
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=bool(cfg["embedding"]["trust_remote_code"])
    ).to(device)
    model.eval()

    batch_size = int(cfg["embedding"]["batch_size"])
    max_seq_len = int(cfg["embedding"]["max_seq_len"])
    normalize = bool(cfg["embedding"]["normalize"])

    count = 0
    dim = None
    with embeddings_path.open("w", encoding="utf-8") as out_f:
        for batch in batched(rows, batch_size):
            texts = [str(x["chunk_text"]) for x in batch]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            pooled = pooled.detach().cpu().float()
            if dim is None:
                dim = int(pooled.shape[1])

            for row, vector in zip(batch, pooled, strict=True):
                out = {"chunk_id": row["chunk_id"], "embedding": vector.tolist()}
                out_f.write(json.dumps(out) + "\n")
                count += 1

    print(f"EMBEDDING_COUNT={count}")
    print(f"EMBEDDING_DIM={dim if dim is not None else 0}")
    print(f"EMBEDDINGS_PATH={embeddings_path}")


if __name__ == "__main__":
    main()
