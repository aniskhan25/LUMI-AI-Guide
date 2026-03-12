#!/usr/bin/env python3
"""Validate required sections for Lesson 09 revised architecture brief."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_SECTIONS = [
    "## 1) Baseline Architecture",
    "## 2) Sensitivity and Trust Context",
    "## 3) Sensitive Stage Identification",
    "## 4) Redesign Decisions",
    "## 5) Revised Data Flow",
    "## 6) Compute Placement and Boundaries",
    "## 7) Trust Gate",
    "## 8) Pilot Scope and Constraints",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", type=Path, required=True, help="Path to revised architecture brief markdown file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.brief.is_file():
        raise SystemExit(f"Brief file not found: {args.brief}")

    text = args.brief.read_text(encoding="utf-8")
    missing = [x for x in REQUIRED_SECTIONS if x not in text]
    if missing:
        print("VALIDATION_OK=0")
        print("Missing sections:")
        for m in missing:
            print(f"- {m}")
        raise SystemExit(1)

    print("VALIDATION_OK=1")
    print(f"All required sections found in: {args.brief}")


if __name__ == "__main__":
    main()

