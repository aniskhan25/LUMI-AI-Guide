# Expected Schemas

## Requests (`sample_requests.jsonl`)

Required fields:

- `request_id`
- `prompt`
- `metadata` (object, optional)

## Responses (`responses.jsonl`)

Required fields:

- `request_id`
- `status` (`ok`)
- `output_text`
- `latency_ms`
- `batch_id`
- `start_ts`
- `end_ts`

## Errors (`errors.jsonl`)

Required fields:

- `request_id`
- `status` (`error`)
- `error_type`
- `error_message`

## Run metadata (`run_metadata.json`)

Required fields:

- `run_name`
- `model_id`
- `gpu_visible_count`
- `mode`
- `batch_size`
- `concurrency`
- `request_count`

## Metrics/summary (`metrics.json`, `summary.json`)

Required fields:

- `processed_count`
- `error_count`
- `p50_latency_ms`
- `p95_latency_ms`
- `throughput_rps`
- `completion_rate`

