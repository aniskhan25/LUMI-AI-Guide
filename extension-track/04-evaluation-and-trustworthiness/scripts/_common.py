#!/usr/bin/env python3
"""Shared helpers for Lesson 04."""

import json
from pathlib import Path

import yaml


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def resolve_path(base_dir, raw_path):
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def resolve_run_dir(cfg, config_path):
    run_dir = resolve_path(config_path.parent, str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
