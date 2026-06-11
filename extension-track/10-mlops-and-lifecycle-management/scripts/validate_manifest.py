#!/usr/bin/env python3
"""Validate required fields for Lesson 10 run manifests."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to run manifest YAML file")
    return parser.parse_args()


def _coerce_scalar(value):
    v = value.strip()
    if v in {"null", "None", "~"}:
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    return v.strip('"').strip("'")


def parse_simple_yaml(text):
    data = {}
    current_section = None
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


def get_nested(data, path):
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def load_manifest(path):
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return parse_simple_yaml(text)


def validate_manifest(data):
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

    missing = []
    for key in required:
        ok, value = get_nested(data, key)
        if not ok or value in {None, ""}:
            missing.append(key)

    return missing


def main():
    args = parse_args()
    from pathlib import Path

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    data = load_manifest(manifest)
    missing = validate_manifest(data)

    if missing:
        print("VALIDATION_OK=0")
        print("Missing required fields:")
        for field in missing:
            print(f"- {field}")
        raise SystemExit(1)

    print("VALIDATION_OK=1")
    print(f"Manifest is complete: {manifest}")


if __name__ == "__main__":
    main()
