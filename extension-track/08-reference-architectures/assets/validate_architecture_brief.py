#!/usr/bin/env python3
"""Validate required sections in an architecture brief markdown file."""

import argparse


REQUIRED_SECTIONS = [
    "## 1) Use Case",
    "## 2) Constraints",
    "## 3) Chosen Reference Architecture",
    "## 4) Data Flow",
    "## 5) Compute Placement",
    "## 6) Operational Checkpoints",
    "## 7) Evaluation Gate",
    "## 8) Pilot Boundary",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", required=True, help="Path to architecture brief markdown file")
    return parser.parse_args()


def main():
    args = parse_args()
    from pathlib import Path

    brief = Path(args.brief)
    if not brief.is_file():
        raise SystemExit(f"Brief file not found: {args.brief}")

    text = brief.read_text(encoding="utf-8")
    missing = [section for section in REQUIRED_SECTIONS if section not in text]

    if missing:
        print("VALIDATION_OK=0")
        print("Missing required sections:")
        for m in missing:
            print(f"- {m}")
        raise SystemExit(1)

    print("VALIDATION_OK=1")
    print(f"Brief contains all required sections: {brief}")


if __name__ == "__main__":
    main()
