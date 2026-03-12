#!/usr/bin/env python3
"""Run service-style inference loop inside one scheduled allocation."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any, Dict, List

from _common import (
    detect_gpu,
    getenv_rank_context,
    load_yaml,
    maybe_gpu_compute,
    now_ms,
    read_jsonl,
    resolve_path,
    resolve_run_dir,
    synthetic_infer,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    return parser.parse_args()


def chunk(rows: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    return [rows[i : i + n] for i in range(0, len(rows), n)]


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config, args.output_root, args.run_name)

    req_path = resolve_path(args.config.parent, str(cfg["paths"]["requests_jsonl"]))
    requests = read_jsonl(req_path)
    if not requests:
        raise SystemExit(f"No requests found in {req_path}")

    service_cfg = cfg["service"]
    batch_size = args.batch_size or int(service_cfg["batch_size"])
    concurrency = args.concurrency or int(service_cfg["concurrency"])
    poll_interval_ms = int(service_cfg["poll_interval_ms"])
    max_new_tokens = int(service_cfg["max_new_tokens"])
    model_id = str(cfg["model"]["model_id"])
    mode = str(cfg["model"]["mode"])

    runtime = cfg["runtime"]
    gpu = detect_gpu(bool(runtime["require_gpu"]), bool(runtime["allow_cpu_fallback"]))
    device = str(gpu["device"])

    requests_copy = run_dir / str(cfg["output"]["requests_copy_jsonl"])
    responses_path = run_dir / str(cfg["output"]["responses_jsonl"])
    errors_path = run_dir / str(cfg["output"]["errors_jsonl"])
    metadata_path = run_dir / str(cfg["output"]["run_metadata_json"])
    write_jsonl(requests_copy, requests)

    queue_batches = chunk(requests, batch_size)
    worker_cycle = itertools.cycle(range(concurrency))
    responses: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    batch_id = 0
    for batch in queue_batches:
        batch_id += 1
        worker_id = next(worker_cycle)
        maybe_gpu_compute(len(batch), device)
        time.sleep(max(0, poll_interval_ms) / 1000.0)

        for row in batch:
            req_id = str(row.get("request_id", "")).strip()
            prompt = str(row.get("prompt", ""))
            start = now_ms()
            if not req_id:
                errors.append(
                    {
                        "request_id": "",
                        "status": "error",
                        "error_type": "schema_error",
                        "error_message": "missing request_id",
                    }
                )
                continue
            try:
                out = synthetic_infer(prompt, max_new_tokens=max_new_tokens)
                if not out:
                    raise ValueError("empty_output")
                end = now_ms()
                responses.append(
                    {
                        "request_id": req_id,
                        "status": "ok",
                        "output_text": out,
                        "latency_ms": end - start,
                        "batch_id": batch_id,
                        "worker_id": worker_id,
                        "start_ts": start,
                        "end_ts": end,
                        "metadata": row.get("metadata", {}),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "request_id": req_id,
                        "status": "error",
                        "error_type": "inference_error",
                        "error_message": str(exc),
                    }
                )

    write_jsonl(responses_path, responses)
    write_jsonl(errors_path, errors)

    metadata = {
        "run_name": run_dir.name,
        "model_id": model_id,
        "mode": mode,
        "pattern": "service",
        "batch_size": batch_size,
        "concurrency": concurrency,
        "poll_interval_ms": poll_interval_ms,
        "request_count": len(requests),
        "response_count": len(responses),
        "error_count": len(errors),
        "gpu_visible_count": int(gpu["gpu_visible_count"]),
        **getenv_rank_context(),
    }
    write_json(metadata_path, metadata)

    print(f"REQUESTS={len(requests)}")
    print(f"RESPONSES={len(responses)}")
    print(f"ERRORS={len(errors)}")
    print(f"RESPONSES_PATH={responses_path}")


if __name__ == "__main__":
    main()

