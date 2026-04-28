#!/usr/bin/env python3
"""Minimal foundation-model adaptation example for Lesson 01."""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:
    raise SystemExit(
        "transformers is required. Use the AI Factory full container or install transformers."
    ) from exc


@dataclass
class Example:
    text: str
    label: int


class JsonlTextDataset(Dataset):
    def __init__(self, records, tokenizer, max_seq_len):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        encoded = self.tokenizer(
            row.text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(row.label, dtype=torch.long),
        }


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


def read_jsonl(path, text_key, label_key, limit):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(Example(text=str(row[text_key]), label=int(row[label_key])))
            if limit > 0 and len(items) >= limit:
                break
    return items


def build_model(cfg):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=int(cfg["model"]["num_labels"]),
    )
    mode = str(cfg["adaptation"]["mode"]).lower()

    if mode == "head_only":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    elif mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise SystemExit(
                "adaptation.mode=lora requires `peft`. Use AI Factory full container or choose head_only/full."
            ) from exc

        lora_cfg = LoraConfig(
            r=int(cfg["adaptation"]["lora_r"]),
            lora_alpha=int(cfg["adaptation"]["lora_alpha"]),
            lora_dropout=float(cfg["adaptation"]["lora_dropout"]),
            task_type=TaskType.SEQ_CLS,
            target_modules=["q_lin", "k_lin", "v_lin"],
        )
        model = get_peft_model(model, lora_cfg)
    elif mode == "full":
        pass
    else:
        raise SystemExit(f"Unsupported adaptation.mode: {mode}")

    return model


def evaluate(model, data_loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            labels = batch["labels"]
            total += labels.numel()
            correct += (preds == labels).sum().item()
            loss_sum += float(outputs.loss.item())

    avg_loss = loss_sum / max(1, len(data_loader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def main():
    args = parse_args()
    cfg = load_config(args.config)

    run_name = args.run_name or str(cfg["run"]["run_name"])
    out_root = Path(str(cfg["run"]["output_dir"])) / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_root / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["run"]["seed"]))

    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"GPU_VISIBLE_COUNT={gpu_visible_count}")

    if not torch.cuda.is_available() and not bool(cfg["runtime"]["allow_cpu_fallback"]):
        raise SystemExit("CUDA device not visible. Set runtime.allow_cpu_fallback=true only for local debugging.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg["model"]["name"]))
    model = build_model(cfg).to(device)

    train_rows = read_jsonl(
        Path(str(cfg["data"]["train_jsonl"])),
        str(cfg["data"]["text_key"]),
        str(cfg["data"]["label_key"]),
        int(cfg["training"]["max_train_samples"]),
    )
    eval_rows = read_jsonl(
        Path(str(cfg["data"]["eval_jsonl"])),
        str(cfg["data"]["text_key"]),
        str(cfg["data"]["label_key"]),
        int(cfg["training"]["max_eval_samples"]),
    )

    train_ds = JsonlTextDataset(train_rows, tokenizer, int(cfg["data"]["max_seq_len"]))
    eval_ds = JsonlTextDataset(eval_rows, tokenizer, int(cfg["data"]["max_seq_len"]))

    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    model.train()
    epochs = int(cfg["training"]["num_epochs"])
    log_every = int(cfg["run"]["log_every_steps"])
    step = 0
    for epoch in range(epochs):
        print(f"EPOCH_START={epoch + 1}")
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if step % log_every == 0:
                print(f"TRAIN_STEP={step} TRAIN_LOSS={loss.item():.6f}")

    eval_loss, eval_acc = evaluate(model, eval_loader, device)
    print(f"EVAL_LOSS={eval_loss:.6f}")
    print(f"EVAL_ACCURACY={eval_acc:.6f}")

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    metrics = {
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "num_train_samples": len(train_ds),
        "num_eval_samples": len(eval_ds),
        "adaptation_mode": str(cfg["adaptation"]["mode"]),
    }
    with (out_root / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "run_name": run_name,
        "model_name": str(cfg["model"]["name"]),
        "device": str(device),
        "gpu_visible_count": gpu_visible_count,
        "output_dir": str(out_root),
    }
    with (out_root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("RUN_COMPLETE=1")


if __name__ == "__main__":
    main()
