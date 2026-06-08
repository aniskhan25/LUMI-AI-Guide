"""
Fine-tune a pretrained HuggingFace model on a text classification task.

Adaptation modes (set via --mode):
  head_only  - freeze base model, train classifier head only (fastest, lowest risk)
  lora       - train small adapter layers via PEFT (good balance)
  full       - update all parameters (most flexible, highest cost)

Default: head_only on distilbert-base-uncased with AG News (4 classes).
"""

import os
import argparse
import json
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["head_only", "lora", "full"], default="head_only")
parser.add_argument("--model", default="distilbert-base-uncased")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--output_dir", default="outputs/finetune")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Dataset ---
print("Loading AG News dataset...")
dataset = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained(args.model)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"].select(range(8000)), batch_size=args.batch_size, shuffle=True)
eval_loader  = DataLoader(dataset["test"].select(range(2000)),  batch_size=args.batch_size)

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=4)

if args.mode == "head_only":
    for name, param in model.base_model.named_parameters():
        param.requires_grad = False
    print("Mode: head_only — base model frozen, training classifier only")

elif args.mode == "lora":
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, config)
        print("Mode: lora — training adapter layers only")
        model.print_trainable_parameters()
    except ImportError:
        print("peft not available — falling back to head_only")
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False

elif args.mode == "full":
    print("Mode: full — training all parameters")

model.to(device)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# --- Training ---
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# --- Evaluation ---
model.eval()
correct = total = 0
with torch.no_grad():
    for batch in eval_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Eval accuracy: {accuracy:.4f}")

# --- Save ---
model.save_pretrained(os.path.join(args.output_dir, "checkpoint"))
tokenizer.save_pretrained(os.path.join(args.output_dir, "checkpoint"))

metrics = {"eval_accuracy": accuracy, "mode": args.mode, "model": args.model}
with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"RUN_COMPLETE=1")
print(f"Checkpoint saved to {args.output_dir}/checkpoint")
