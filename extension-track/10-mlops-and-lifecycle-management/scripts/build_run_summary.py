#!/usr/bin/env python3
"""Build a compact lifecycle summary from a Lesson 10 run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="Path to run manifest YAML file")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown summary path")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON summary path")
    return parser.parse_args()


def _coerce_scalar(value: str) -> Any:
    v = value.strip()
    if v in {"null", "None", "~"}:
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    return v.strip('"').strip("'")


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_section: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if indent == 0:
            if not value:
                current_section = key
                data.setdefault(key, {})
            else:
                current_section = None
                data[key] = _coerce_scalar(value)
        elif indent == 2 and current_section:
            section = data.setdefault(current_section, {})
            if isinstance(section, dict):
                section[key] = _coerce_scalar(value)

    return data


def get_nested(data: Dict[str, Any], path: str, default: Any = "") -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_manifest(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return parse_simple_yaml(text)


def completeness(data: Dict[str, Any]) -> Tuple[int, int, List[str]]:
    required = [
        "run_id",
        "lifecycle_state",
        "owner",
        "versions.dataset_version",
        "versions.config_version",
        "versions.model_or_adapter_ref",
        "versions.container_ref",
        "paths.input_path",
        "paths.output_path",
        "evaluation.benchmark_id",
        "promotion.status",
    ]
    missing: List[str] = []
    for field in required:
        value = get_nested(data, field, default=None)
        if value in {None, ""}:
            missing.append(field)
    return len(required) - len(missing), len(required), missing


def build_markdown(data: Dict[str, Any], score: Tuple[int, int, List[str]]) -> str:
    passed, total, missing = score

    run_id = get_nested(data, "run_id")
    state = get_nested(data, "lifecycle_state")
    owner = get_nested(data, "owner")

    dataset = get_nested(data, "versions.dataset_version")
    config = get_nested(data, "versions.config_version")
    model = get_nested(data, "versions.model_or_adapter_ref")
    container = get_nested(data, "versions.container_ref")

    in_path = get_nested(data, "paths.input_path")
    out_path = get_nested(data, "paths.output_path")
    promoted_path = get_nested(data, "paths.promoted_path")
    share_path = get_nested(data, "paths.share_path")

    bench = get_nested(data, "evaluation.benchmark_id")
    gate = get_nested(data, "evaluation.summary.pass_gate")
    promotion = get_nested(data, "promotion.status")

    lines: List[str] = []
    lines.append("# Run Lifecycle Summary")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- lifecycle_state: `{state}`")
    lines.append(f"- owner: `{owner}`")
    lines.append(f"- completeness: `{passed}/{total}`")
    lines.append("")
    lines.append("## Versioned Inputs")
    lines.append("")
    lines.append(f"- dataset_version: `{dataset}`")
    lines.append(f"- config_version: `{config}`")
    lines.append(f"- model_or_adapter_ref: `{model}`")
    lines.append(f"- container_ref: `{container}`")
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    lines.append(f"- input_path: `{in_path}`")
    lines.append(f"- output_path: `{out_path}`")
    lines.append(f"- promoted_path: `{promoted_path}`")
    lines.append(f"- share_path: `{share_path}`")
    lines.append("")
    lines.append("## Evaluation and Promotion")
    lines.append("")
    lines.append(f"- benchmark_id: `{bench}`")
    lines.append(f"- pass_gate: `{gate}`")
    lines.append(f"- promotion_status: `{promotion}`")

    if missing:
        lines.append("")
        lines.append("## Missing Required Fields")
        lines.append("")
        for field in missing:
            lines.append(f"- `{field}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    data = load_manifest(args.manifest)
    score = completeness(data)
    summary_md = build_markdown(data, score)

    output_md = args.output or (args.manifest.parent / "run-summary.md")
    output_md.write_text(summary_md, encoding="utf-8")

    output_json = args.json_output or (args.manifest.parent / "run-summary.json")
    output_payload = {
        "run_id": get_nested(data, "run_id", ""),
        "lifecycle_state": get_nested(data, "lifecycle_state", ""),
        "owner": get_nested(data, "owner", ""),
        "completeness_passed": score[0],
        "completeness_total": score[1],
        "missing_fields": score[2],
        "promotion_status": get_nested(data, "promotion.status", ""),
    }
    output_json.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")

    print(f"SUMMARY_MD={output_md}")
    print(f"SUMMARY_JSON={output_json}")


if __name__ == "__main__":
    main()
