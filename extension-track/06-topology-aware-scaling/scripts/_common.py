#!/usr/bin/env python3
"""Shared helpers for Lesson 06."""

import json
import os
from pathlib import Path

import yaml


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir, raw_path):
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def resolve_run_dir(cfg, cfg_path):
    run_dir = resolve_path(cfg_path.parent, str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def rank_info():
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(path, prefix):
    if not path.is_dir():
        return []
    return sorted([file for file in path.glob(f"{prefix}*.json") if file.is_file()])
