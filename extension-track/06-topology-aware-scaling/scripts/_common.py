#!/usr/bin/env python3
"""Shared helpers for Lesson 06 scaling scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

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


def rank_info() -> Dict[str, Any]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(path: Path, prefix: str) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted([p for p in path.glob(f"{prefix}*.json") if p.is_file()])

