#!/usr/bin/env python3
"""Validate required sections and ownership fields for Lesson 11 templates."""

import argparse


REQUIRED_BLUEPRINT_SECTIONS = [
    "## 1) Project Context",
    "## 2) Team Roles",
    "## 3) Artifact Classes and Owners",
    "## 4) Storage and Sharing Boundaries",
    "## 5) Review and Promotion Rules",
    "## 6) Handoff Contract",
]

REQUIRED_MATRIX_HEADERS = [
    "| Task | Data Custodian | Workflow Developer | Evaluator | Promotion Approver | Delivery Owner |",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", required=True, help="Path to team-operating-model markdown")
    parser.add_argument("--matrix", required=True, help="Path to responsibility-matrix markdown")
    return parser.parse_args()


def check_required(path, required):
    text = path.read_text(encoding="utf-8")
    return [x for x in required if x not in text]


def main():
    args = parse_args()
    from pathlib import Path

    blueprint = Path(args.blueprint)
    matrix = Path(args.matrix)

    if not blueprint.is_file():
        raise SystemExit(f"Blueprint file not found: {args.blueprint}")
    if not matrix.is_file():
        raise SystemExit(f"Matrix file not found: {args.matrix}")

    missing_blueprint = check_required(blueprint, REQUIRED_BLUEPRINT_SECTIONS)
    missing_matrix = check_required(matrix, REQUIRED_MATRIX_HEADERS)

    if missing_blueprint or missing_matrix:
        print("VALIDATION_OK=0")
        if missing_blueprint:
            print("Missing blueprint sections:")
            for item in missing_blueprint:
                print(f"- {item}")
        if missing_matrix:
            print("Missing matrix headers:")
            for item in missing_matrix:
                print(f"- {item}")
        raise SystemExit(1)

    print("VALIDATION_OK=1")
    print(f"Blueprint and matrix look complete: {blueprint} | {matrix}")


if __name__ == "__main__":
    main()
