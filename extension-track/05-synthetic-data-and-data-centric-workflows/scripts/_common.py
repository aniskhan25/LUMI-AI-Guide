#!/usr/bin/env python3
"""Shared helpers for Lesson 05."""

import json
import re
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


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 %.-]", "", text)
    return text


def qa_key(question, answer):
    return f"{normalize_text(question)}||{normalize_text(answer)}"


def term_overlap_score(text, required_terms):
    if not required_terms:
        return 0.0
    text_l = text.lower()
    hits = sum(1 for term in required_terms if str(term).lower() in text_l)
    return hits / len(required_terms)
