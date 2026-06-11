#!/usr/bin/env python3
"""Generate chunk embeddings for Lesson 03."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModel, AutoTokenizer


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


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = run_dir_from(cfg)

    chunks_path = run_dir / str(cfg["output"]["chunks_jsonl"])
    embeddings_path = run_dir / str(cfg["output"]["embeddings_jsonl"])
    rows = read_jsonl(chunks_path)
    if not rows:
        raise SystemExit(f"No chunks found at {chunks_path}. Run chunk_corpus.py first.")

    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"GPU_VISIBLE_COUNT={gpu_visible_count}")
    if not torch.cuda.is_available() and not bool(cfg["runtime"]["allow_cpu_fallback"]):
        raise SystemExit("CUDA device not visible. Set runtime.allow_cpu_fallback=true only for local debugging.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = str(cfg["embedding"]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(cfg["embedding"]["trust_remote_code"]))
    model = AutoModel.from_pretrained(model_name, trust_remote_code=bool(cfg["embedding"]["trust_remote_code"])).to(device)
    model.eval()

    batch_size = int(cfg["embedding"]["batch_size"])
    max_seq_len = int(cfg["embedding"]["max_seq_len"])
    normalize = bool(cfg["embedding"]["normalize"])

    count = 0
    dim = None
    with embeddings_path.open("w", encoding="utf-8") as out_f:
        for batch in batched(rows, batch_size):
            texts = [str(row["chunk_text"]) for row in batch]
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
                out_f.write(json.dumps({"chunk_id": row["chunk_id"], "embedding": vector.tolist()}) + "\n")
                count += 1

    print(f"EMBEDDING_COUNT={count}")
    print(f"EMBEDDING_DIM={dim if dim is not None else 0}")
    print(f"EMBEDDINGS_PATH={embeddings_path}")


if __name__ == "__main__":
    main()
