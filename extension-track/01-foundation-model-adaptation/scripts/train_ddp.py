#!/usr/bin/env python3
"""Single-node DDP adaptation run for Lesson 01."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from train import JsonlTextDataset, build_model, load_config, read_jsonl, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="baseline-run-ddp")
    return parser.parse_args()


def evaluate(model, data_loader, device):
    model.eval()
    total = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    loss_sum = torch.zeros(1, device=device)

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            total += labels.numel()
            correct += (preds == labels).sum()
            loss_sum += outputs.loss.detach()

    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

    avg_loss = loss_sum.item() / max(1, len(data_loader) * dist.get_world_size())
    accuracy = correct.item() / max(1, total.item())
    return avg_loss, accuracy


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    set_seed(int(cfg["run"]["seed"]) + rank)

    run_name = args.run_name
    out_root = Path(str(cfg["run"]["output_dir"])) / run_name
    checkpoint_dir = out_root / "checkpoint"
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    gpu_visible_count = torch.cuda.device_count()
    if rank == 0:
        print(f"GPU_VISIBLE_COUNT={gpu_visible_count}")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg["model"]["name"]))
    model = build_model(cfg).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])

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

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    eval_sampler = DistributedSampler(eval_ds, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        sampler=train_sampler,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        sampler=eval_sampler,
    )

    opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    epochs = int(cfg["training"]["num_epochs"])
    log_every = int(cfg["run"]["log_every_steps"])
    step = 0
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if rank == 0 and step % log_every == 0:
                print(f"TRAIN_STEP={step} TRAIN_LOSS={loss.item():.6f}")

    eval_loss, eval_acc = evaluate(model, eval_loader, device)
    if rank == 0:
        print(f"EVAL_LOSS={eval_loss:.6f}")
        print(f"EVAL_ACCURACY={eval_acc:.6f}")

        model.module.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        metrics = {
            "eval_loss": eval_loss,
            "eval_accuracy": eval_acc,
            "num_train_samples": len(train_ds),
            "num_eval_samples": len(eval_ds),
            "adaptation_mode": str(cfg["adaptation"]["mode"]),
            "world_size": world_size,
        }
        with (out_root / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        summary = {
            "run_name": run_name,
            "model_name": str(cfg["model"]["name"]),
            "device": "ddp",
            "gpu_visible_count": gpu_visible_count,
            "output_dir": str(out_root),
            "world_size": world_size,
        }
        with (out_root / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("RUN_COMPLETE=1")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
