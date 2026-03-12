#!/usr/bin/env python3
"""Validate required sections for Lesson 13 domain templates."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_USE_CASE_SECTIONS = [
    "## 1) Use Case Definition",
    "## 2) Knowledge Source Scope",
    "## 3) Workflow Goal",
    "## 4) Architecture Choice",
    "## 5) Success Criteria",
]

REQUIRED_SCHEMA_HEADERS = [
    "## Document-Level Fields",
    "## Chunk-Level Fields",
    "## Rules",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-case", type=Path, required=True, help="Path to domain use-case brief markdown")
    parser.add_argument("--schema", type=Path, required=True, help="Path to technical corpus schema markdown")
    return parser.parse_args()


def find_missing(path: Path, required: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [entry for entry in required if entry not in text]


def main() -> None:
    args = parse_args()

    if not args.use_case.is_file():
        raise SystemExit(f"Use-case file not found: {args.use_case}")
    if not args.schema.is_file():
        raise SystemExit(f"Schema file not found: {args.schema}")

    missing_use_case = find_missing(args.use_case, REQUIRED_USE_CASE_SECTIONS)
    missing_schema = find_missing(args.schema, REQUIRED_SCHEMA_HEADERS)

    if missing_use_case or missing_schema:
        print("VALIDATION_OK=0")
        if missing_use_case:
            print("Missing use-case sections:")
            for item in missing_use_case:
                print(f"- {item}")
        if missing_schema:
            print("Missing schema sections:")
            for item in missing_schema:
                print(f"- {item}")
        raise SystemExit(1)

    print("VALIDATION_OK=1")
    print(f"Domain templates are complete: {args.use_case} | {args.schema}")


if __name__ == "__main__":
    main()
