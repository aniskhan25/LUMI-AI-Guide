#!/usr/bin/env python3
"""Run retrieval and grounded answer generation for Lesson 03."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


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


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts, tokenizer, model, device, max_seq_len, normalize):
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
    return pooled.detach().cpu().float()


def build_prompt(query, retrieved):
    lines = [
        "Use the evidence passages to answer the question.",
        "If the evidence is insufficient, say you do not have enough evidence.",
        "",
        f"Question: {query}",
        "",
        "Evidence:",
    ]
    for i, row in enumerate(retrieved, start=1):
        lines.append(f"[{i}] ({row['chunk_id']}) {row['chunk_text']}")
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def fallback_answer(retrieved):
    if not retrieved:
        return "I do not have enough evidence in the retrieved context."
    return f"Based on retrieved evidence: {retrieved[0]['chunk_text'].strip()}"


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = run_dir_from(cfg)

    chunks_path = run_dir / str(cfg["output"]["chunks_jsonl"])
    index_path = run_dir / str(cfg["output"]["retriever_index_npz"])
    retrieval_path = run_dir / str(cfg["output"]["retrieval_results_jsonl"])
    answers_path = run_dir / str(cfg["output"]["answers_jsonl"])
    summary_path = run_dir / str(cfg["output"]["summary_json"])

    queries_path = Path(str(cfg["data"]["queries_jsonl"]))
    queries = read_jsonl(queries_path)
    if not queries:
        raise SystemExit(f"No queries found in {queries_path}")

    chunks = read_jsonl(chunks_path)
    chunk_by_id = {str(row["chunk_id"]): row for row in chunks}

    index_data = np.load(index_path, allow_pickle=False)
    index_chunk_ids = [str(x) for x in index_data["chunk_ids"]]
    emb_matrix = np.asarray(index_data["embeddings"], dtype=np.float32)
    if emb_matrix.ndim != 2:
        raise SystemExit(f"Invalid embedding matrix shape in {index_path}: {emb_matrix.shape}")

    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"GPU_VISIBLE_COUNT={gpu_visible_count}")
    if not torch.cuda.is_available() and not bool(cfg["runtime"]["allow_cpu_fallback"]):
        raise SystemExit("CUDA device not visible. Set runtime.allow_cpu_fallback=true only for local debugging.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_model_name = str(cfg["embedding"]["model_name"])
    emb_tok = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=bool(cfg["embedding"]["trust_remote_code"]))
    emb_model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=bool(cfg["embedding"]["trust_remote_code"])).to(device)
    emb_model.eval()

    query_key = str(cfg["data"]["query_text_key"])
    query_id_key = str(cfg["data"]["query_id_key"])
    max_seq_len = int(cfg["embedding"]["max_seq_len"])
    normalize = bool(cfg["embedding"]["normalize"])
    top_k = int(cfg["retrieval"]["top_k"])

    start = time.time()
    retrieval_rows = []
    for query in queries:
        query_id = str(query[query_id_key])
        query_text = str(query[query_key])
        query_vector = encode_texts([query_text], emb_tok, emb_model, device, max_seq_len, normalize).numpy()[0]
        scores = emb_matrix @ query_vector
        top_idx = np.argsort(-scores)[:top_k]
        retrieved = []
        for idx in top_idx:
            chunk_id = index_chunk_ids[int(idx)]
            chunk = chunk_by_id.get(chunk_id, {})
            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": chunk.get("doc_id", ""),
                    "score": float(scores[int(idx)]),
                    "chunk_text": chunk.get("chunk_text", ""),
                }
            )
        retrieval_rows.append({"query_id": query_id, "query": query_text, "retrieved": retrieved})

    with retrieval_path.open("w", encoding="utf-8") as f:
        for row in retrieval_rows:
            f.write(json.dumps(row) + "\n")

    generation_model_name = str(cfg["generation"]["model_name"])
    generation_backend = "hf_causal"
    try:
        gen_tok = AutoTokenizer.from_pretrained(generation_model_name, trust_remote_code=bool(cfg["generation"]["trust_remote_code"]))
        if gen_tok.pad_token is None:
            gen_tok.pad_token = gen_tok.eos_token
        gen_model = AutoModelForCausalLM.from_pretrained(generation_model_name, trust_remote_code=bool(cfg["generation"]["trust_remote_code"])).to(device)
        gen_model.eval()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: generation model unavailable ({exc}); using fallback templated answers.")
        generation_backend = "fallback_template"
        gen_tok = None
        gen_model = None

    max_input = int(cfg["generation"]["max_input_length"])
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    do_sample = bool(cfg["generation"]["do_sample"])
    temperature = float(cfg["generation"]["temperature"])
    top_p = float(cfg["generation"]["top_p"])

    answer_rows = []
    for row in retrieval_rows:
        prompt = build_prompt(row["query"], row["retrieved"])
        if generation_backend == "hf_causal" and gen_tok is not None and gen_model is not None:
            encoded = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=max_input)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                generated = gen_model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=gen_tok.eos_token_id,
                )
            prompt_len = encoded["input_ids"].shape[1]
            answer = gen_tok.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
            if not answer:
                answer = fallback_answer(row["retrieved"])
        else:
            answer = fallback_answer(row["retrieved"])

        answer_rows.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "answer": answer,
                "evidence_chunk_ids": [item["chunk_id"] for item in row["retrieved"]],
                "generation_backend": generation_backend,
            }
        )

    with answers_path.open("w", encoding="utf-8") as f:
        for row in answer_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "mode": "rag",
        "run_name": str(cfg["run"]["run_name"]),
        "device": str(device),
        "gpu_visible_count": gpu_visible_count,
        "embedding_model": embedding_model_name,
        "generation_model": generation_model_name,
        "generation_backend": generation_backend,
        "top_k": top_k,
        "query_count": len(queries),
        "chunk_count": len(chunks),
        "retrieval_results_path": str(retrieval_path),
        "answers_path": str(answers_path),
        "elapsed_seconds": time.time() - start,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"RETRIEVAL_COUNT={len(retrieval_rows)}")
    print(f"ANSWER_COUNT={len(answer_rows)}")
    print("RUN_COMPLETE=1")
    print(f"RETRIEVAL_RESULTS={retrieval_path}")
    print(f"ANSWERS_PATH={answers_path}")
    print(f"SUMMARY_PATH={summary_path}")


if __name__ == "__main__":
    main()
