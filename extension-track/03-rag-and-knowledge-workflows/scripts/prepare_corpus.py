#!/usr/bin/env python3
"""Wrapper to run corpus preparation from scripts/ path."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "data" / "prepare_corpus.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

