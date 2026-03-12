#!/usr/bin/env python3
"""Validate required sections for Lesson 12 planning templates."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_PROFILE_SECTIONS = [
    "## 1) Workload Identity",
    "## 2) Input and Output Shape",
    "## 3) Baseline Configuration",
    "## 4) Target Configuration",
    "## 5) Scale-Up Gate",
    "## 6) Artifact Reuse Strategy",
    "## 7) Stop Criteria",
]

REQUIRED_RUN_PLAN_HEADERS = [
    "| Stage | Purpose | Partition | Resources | Walltime | Expected Output | Decision Gate |",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True, help="Path to workload-profile markdown")
    parser.add_argument("--run-plan", type=Path, required=True, help="Path to staged-run-plan markdown")
    return parser.parse_args()


def missing_items(path: Path, required: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [item for item in required if item not in text]


def main() -> None:
    args = parse_args()

    if not args.profile.is_file():
        raise SystemExit(f"Profile file not found: {args.profile}")
    if not args.run_plan.is_file():
        raise SystemExit(f"Run plan file not found: {args.run_plan}")

    missing_profile = missing_items(args.profile, REQUIRED_PROFILE_SECTIONS)
    missing_plan = missing_items(args.run_plan, REQUIRED_RUN_PLAN_HEADERS)

    if missing_profile or missing_plan:
        print("VALIDATION_OK=0")
        if missing_profile:
            print("Missing profile sections:")
            for item in missing_profile:
                print(f"- {item}")
        if missing_plan:
            print("Missing run-plan headers:")
            for item in missing_plan:
                print(f"- {item}")
        raise SystemExit(1)

    print("VALIDATION_OK=1")
    print(f"Planning templates are complete: {args.profile} | {args.run_plan}")


if __name__ == "__main__":
    main()
