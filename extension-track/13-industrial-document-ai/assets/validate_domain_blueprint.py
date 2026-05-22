#!/usr/bin/env python3
"""Validate required sections for Lesson 13 domain templates."""

import argparse


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-case", required=True, help="Path to domain use-case brief markdown")
    parser.add_argument("--schema", required=True, help="Path to technical corpus schema markdown")
    return parser.parse_args()


def find_missing(path, required):
    text = path.read_text(encoding="utf-8")
    return [entry for entry in required if entry not in text]


def main():
    args = parse_args()
    from pathlib import Path

    use_case = Path(args.use_case)
    schema = Path(args.schema)

    if not use_case.is_file():
        raise SystemExit(f"Use-case file not found: {args.use_case}")
    if not schema.is_file():
        raise SystemExit(f"Schema file not found: {args.schema}")

    missing_use_case = find_missing(use_case, REQUIRED_USE_CASE_SECTIONS)
    missing_schema = find_missing(schema, REQUIRED_SCHEMA_HEADERS)

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
    print(f"Domain templates are complete: {use_case} | {schema}")


if __name__ == "__main__":
    main()
