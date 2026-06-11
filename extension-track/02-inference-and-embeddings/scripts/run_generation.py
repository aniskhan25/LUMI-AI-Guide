#!/usr/bin/env python3
"""Optional batched generation pipeline for Lesson 02."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path, max_samples):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    args = parse_args()
    cfg = load_config(args.config)

    run_name = args.run_name or str(cfg["run"]["run_name"])
    out_dir = Path(str(cfg["run"]["output_dir"])) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    input_jsonl = Path(str(cfg["data"]["input_jsonl"]))
    outputs_path = out_dir / str(cfg["output"]["outputs_filename"])
    summary_path = out_dir / str(cfg["output"]["summary_filename"])

    set_seed(int(cfg["run"]["seed"]))

    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"GPU_VISIBLE_COUNT={gpu_visible_count}")

    if not torch.cuda.is_available() and not bool(cfg["runtime"]["allow_cpu_fallback"]):
        raise SystemExit("CUDA device not visible. Set runtime.allow_cpu_fallback=true only for local debugging.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = str(cfg["model"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(cfg["model"]["trust_remote_code"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=bool(cfg["model"]["trust_remote_code"])).to(device)
    model.eval()

    rows = read_jsonl(input_jsonl, int(cfg["data"]["max_samples"]))
    if not rows:
        raise SystemExit(f"No input records found in {input_jsonl}")

    id_key = str(cfg["data"]["id_key"])
    prompt_key = str(cfg["data"]["prompt_key"])
    batch_size = int(cfg["inference"]["batch_size"])
    max_input_length = int(cfg["inference"]["max_input_length"])
    max_new_tokens = int(cfg["inference"]["max_new_tokens"])
    do_sample = bool(cfg["inference"]["do_sample"])
    temperature = float(cfg["inference"]["temperature"])
    top_p = float(cfg["inference"]["top_p"])
    log_every = int(cfg["run"]["log_every_batches"])

    start = time.time()
    total = 0

    with outputs_path.open("w", encoding="utf-8") as out_f:
        for idx, batch_rows in enumerate(batched(rows, batch_size), start=1):
            prompts = [str(r[prompt_key]) for r in batch_rows]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for row, seq, prompt_len in zip(batch_rows, generated, prompt_lengths, strict=True):
                generated_tokens = seq[int(prompt_len) :]
                text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                out_f.write(json.dumps({"id": row[id_key], "prompt": row[prompt_key], "output_text": text}) + "\n")
                total += 1

            if idx % log_every == 0:
                print(f"BATCH={idx} OUTPUT_RECORDS={total}")

    elapsed = time.time() - start
    summary = {
        "mode": "generation",
        "run_name": run_name,
        "model_name": model_name,
        "device": str(device),
        "gpu_visible_count": gpu_visible_count,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(outputs_path),
        "records_written": total,
        "batch_size": batch_size,
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": elapsed,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("RUN_COMPLETE=1")
    print(f"OUTPUT_JSONL={outputs_path}")
    print(f"SUMMARY_JSON={summary_path}")


if __name__ == "__main__":
    main()
