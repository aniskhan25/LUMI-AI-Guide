#!/usr/bin/env python3
"""Validate Lesson 07 outputs."""

from pathlib import Path

from _common import load_yaml, read_json, read_jsonl, resolve_path


def ensure_file(path):
    if not path.is_file():
        raise SystemExit(f"Missing expected file: {path}")


def validate_run(config_path):
    cfg = load_yaml(config_path)
    outputs_dir = resolve_path(config_path.parent, str(cfg["run"]["output_dir"]))
    run_dir = outputs_dir / str(cfg["run"]["run_name"])

    requests_path = run_dir / str(cfg["output"]["requests_copy_jsonl"])
    responses_path = run_dir / str(cfg["output"]["responses_jsonl"])
    errors_path = run_dir / str(cfg["output"]["errors_jsonl"])
    metadata_path = run_dir / str(cfg["output"]["run_metadata_json"])
    metrics_path = run_dir / str(cfg["output"]["metrics_json"])
    summary_path = run_dir / str(cfg["output"]["summary_json"])

    for path in [requests_path, responses_path, errors_path, metadata_path, metrics_path, summary_path]:
        ensure_file(path)

    requests = read_jsonl(requests_path)
    responses = read_jsonl(responses_path)
    errors = read_jsonl(errors_path)
    metadata = read_json(metadata_path)
    summary = read_json(summary_path)

    request_ids = {str(row["request_id"]) for row in requests if str(row.get("request_id", "")).strip()}
    response_ids = {str(row["request_id"]) for row in responses if str(row.get("request_id", "")).strip()}
    error_ids = {str(row["request_id"]) for row in errors if str(row.get("request_id", "")).strip()}

    if request_ids != response_ids | error_ids:
        raise SystemExit(f"Request coverage mismatch in {run_dir}")

    if int(summary.get("request_count", 0)) != len(requests):
        raise SystemExit(f"Request count mismatch in {summary_path}")
    if int(summary.get("response_count", 0)) != len(responses):
        raise SystemExit(f"Response count mismatch in {summary_path}")
    if int(summary.get("error_count", 0)) != len(errors):
        raise SystemExit(f"Error count mismatch in {summary_path}")
    if int(metadata.get("gpu_visible_count", 0)) < 1:
        raise SystemExit(f"No visible GPU recorded in {metadata_path}")

    return run_dir.name, len(requests), len(responses), len(errors), int(metadata.get("gpu_visible_count", 0))


def main():
    lesson_dir = Path(__file__).resolve().parents[1]
    configs_dir = lesson_dir / "configs"
    inference = validate_run(configs_dir / "inference.yaml")
    service = validate_run(configs_dir / "service.yaml")

    outputs_dir = resolve_path(configs_dir, str(load_yaml(configs_dir / "inference.yaml")["run"]["output_dir"]))
    ensure_file(outputs_dir / "advanced-inference-comparison.json")
    ensure_file(outputs_dir / "advanced-inference-comparison.md")

    print("VALIDATION_OK=1")
    print(f"batched_requests={inference[1]}")
    print(f"service_requests={service[1]}")
    print(f"batched_gpu_visible_count={inference[4]}")
    print(f"service_gpu_visible_count={service[4]}")


if __name__ == "__main__":
    main()
