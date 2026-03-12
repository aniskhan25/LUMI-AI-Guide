#!/usr/bin/env python3
"""Shared helpers for Lesson 05 scripts."""

from __future__ import annotations

import json
import re
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


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 %.-]", "", text)
    return text


def qa_key(question: str, answer: str) -> str:
    return f"{normalize_text(question)}||{normalize_text(answer)}"


def term_overlap_score(text: str, required_terms: List[str]) -> float:
    if not required_terms:
        return 0.0
    text_l = text.lower()
    hits = sum(1 for t in required_terms if str(t).lower() in text_l)
    return hits / len(required_terms)

