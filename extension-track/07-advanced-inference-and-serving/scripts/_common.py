#!/usr/bin/env python3
"""Shared helpers for Lesson 07 inference scripts."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml is required. Install PyYAML or run inside the AI Factory container.") from exc


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (base_dir / p).resolve()


def resolve_run_dir(cfg: Dict[str, Any], cfg_path: Path, output_root: Path | None, run_name: str | None) -> Path:
    name = run_name or str(cfg["run"]["run_name"])
    root = output_root or resolve_path(cfg_path.parent, str(cfg["run"]["output_dir"]))
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_gpu(require_gpu: bool, allow_cpu_fallback: bool) -> Dict[str, Any]:
    out = {"gpu_visible_count": 0, "device": "cpu", "torch_available": False}
    try:
        import torch

        out["torch_available"] = True
        if torch.cuda.is_available():
            out["gpu_visible_count"] = torch.cuda.device_count()
            out["device"] = "cuda:0"
        if require_gpu and out["gpu_visible_count"] < 1 and not allow_cpu_fallback:
            raise SystemExit("GPU required but no CUDA device is visible.")
    except ImportError:
        if require_gpu and not allow_cpu_fallback:
            raise SystemExit("torch is required to validate GPU visibility.")
    return out


def now_ms() -> int:
    return int(time.time() * 1000)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def synthetic_infer(prompt: str, max_new_tokens: int) -> str:
    prompt_clean = " ".join(prompt.strip().split())
    if not prompt_clean:
        return ""
    head = prompt_clean[: max_new_tokens * 2]
    return f"Processed response: {head}"


def maybe_gpu_compute(batch_size: int, device: str) -> None:
    if not device.startswith("cuda"):
        time.sleep(0.002 * max(1, batch_size))
        return
    try:
        import torch

        x = torch.randn(batch_size, 1024, device=device)
        w = torch.randn(1024, 1024, device=device)
        y = torch.relu(x @ w)
        _ = float(y.mean().item())
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        # Keep pipeline robust even if GPU op fails unexpectedly.
        time.sleep(0.003 * max(1, batch_size))


def getenv_rank_context() -> Dict[str, Any]:
    return {
        "hostname": os.environ.get("HOSTNAME", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_nodeid": os.environ.get("SLURM_NODEID", ""),
        "slurm_procid": os.environ.get("SLURM_PROCID", ""),
    }
