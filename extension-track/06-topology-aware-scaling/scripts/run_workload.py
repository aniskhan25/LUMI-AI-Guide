#!/usr/bin/env python3
"""Run a compact scaling workload and emit per-rank metrics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

from _common import load_yaml, rank_info, resolve_run_dir, write_json

try:
    import torch
except ImportError as exc:
    raise SystemExit("torch is required. Run inside AI Factory container.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def maybe_init_distributed(world_size: int) -> bool:
    if world_size <= 1:
        return False
    if not torch.distributed.is_available():
        raise SystemExit("torch.distributed is not available in this environment.")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    return True


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config, args.output_root, args.run_name)
    raw_dir = run_dir / str(cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    info = rank_info()
    rank = int(info["rank"])
    local_rank = int(info["local_rank"])
    world_size = int(info["world_size"])

    require_gpu = bool(cfg["runtime"]["require_gpu"])
    allow_cpu_fallback = bool(cfg["runtime"]["allow_cpu_fallback"])
    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if require_gpu and gpu_visible_count < 1 and not allow_cpu_fallback:
        raise SystemExit("GPU required but no CUDA device is visible.")

    if torch.cuda.is_available():
        device_index = local_rank % max(1, gpu_visible_count)
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    is_distributed = maybe_init_distributed(world_size)
    if is_distributed:
        torch.distributed.barrier()

    samples_per_step = int(cfg["workload"]["samples_per_step"])
    steps = int(cfg["workload"]["steps"])
    warmup_steps = int(cfg["workload"]["warmup_steps"])
    hidden = int(cfg["workload"]["hidden_size"])
    repeats = int(cfg["workload"]["compute_repeats"])

    x = torch.randn(samples_per_step, hidden, device=device)
    w = torch.randn(hidden, hidden, device=device)
    b = torch.randn(hidden, device=device)

    t0 = time.perf_counter()
    for step in range(steps + warmup_steps):
        y = x
        for _ in range(repeats):
            y = torch.relu(torch.matmul(y, w) + b)
        if is_distributed:
            # Simulate communication pressure in scaled runs.
            comm_buf = y.mean(dim=0)
            torch.distributed.all_reduce(comm_buf)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    elapsed = max(1e-9, t1 - t0)
    effective_steps = max(1, steps)
    effective_samples = effective_steps * samples_per_step
    throughput = effective_samples / elapsed

    payload: Dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": str(device),
        "gpu_visible_count": gpu_visible_count,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "samples_per_step": samples_per_step,
        "hidden_size": hidden,
        "compute_repeats": repeats,
        "elapsed_seconds": elapsed,
        "throughput_samples_per_sec": throughput,
    }

    out_path = raw_dir / f"{cfg['output']['metrics_prefix']}{rank}.json"
    write_json(out_path, payload)

    if is_distributed:
        torch.distributed.barrier()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    print(f"METRICS_PATH={out_path}")
    print(f"RANK={rank}")
    print(f"THROUGHPUT={throughput:.4f}")


if __name__ == "__main__":
    main()

