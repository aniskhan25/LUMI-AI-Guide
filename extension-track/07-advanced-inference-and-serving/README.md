# 07. Advanced Inference and Serving Patterns on LUMI-G

This lesson teaches how to package model inference into repeatable high-throughput patterns on LUMI-G.

## What This Lesson Enables

Run and compare two inference patterns:

- batched inference for bulk request sets
- service-style request loop inside a scheduled GPU job

Both paths produce structured request/response artifacts and throughput/latency summaries.

## When To Use This Workflow

Use this lesson when:

- you need repeated internal inference at scale
- throughput and response quality both matter
- you must choose batch-style vs service-style execution

Do not use this lesson for:

- public internet-facing APIs
- enterprise gateway/auth architecture
- full cloud-native autoscaling operations

## Prerequisites

- Working LUMI access with AI Factory container setup
- Completion of onboarding lessons
- Preferred: completion of extension Lessons 2, 3, and 6
- Access to this repository and sample request set

## Workflow At A Glance

```mermaid
flowchart LR
  A["Request JSONL"] --> B["Batching / concurrency"]
  B --> C["Inference engine on LUMI-G"]
  C --> D["Response JSONL + errors"]
  D --> E["Latency + throughput metrics"]
  E --> F["Operating summary and comparison"]
```

## Minimal Working Example

Work from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/07-advanced-inference-and-serving
```

1. Run batched inference:

```bash
python scripts/run_batched_inference.py --config configs/inference.yaml
```

2. Run service-style loop:

```bash
python scripts/run_service_loop.py --config configs/service.yaml
```

3. Collect metrics:

```bash
python scripts/collect_metrics.py --config configs/inference.yaml --mode batched
python scripts/collect_metrics.py --config configs/service.yaml --mode service
```

4. Build summary comparison:

```bash
python scripts/summarize_results.py --compare-config configs/compare.yaml
```

5. Canonical Slurm jobs:

```bash
sbatch jobs/run_batched_inference.sh
sbatch jobs/run_service_style_inference.sh
```

## How To Verify It Worked

Confirm all of these:

- model metadata file indicates GPU visibility
- request and response counts match (or explicit error records exist)
- `request_id` joins cleanly across requests/responses/errors
- latency and throughput summaries exist
- comparison report contains at least one controlled configuration delta

Expected outputs: [assets/expected-output-tree.txt](assets/expected-output-tree.txt)

## Which Serving Pattern To Choose

Use batch-style when:

- processing large queued corpora
- maximizing throughput is primary goal

Use service-style loop when:

- repeated internal requests arrive during one allocation
- you need lower turnaround than offline-only batch

Consider cloud-native alternatives when:

- continuously available endpoints are required
- broader web-platform integration is mandatory

## Throughput And Latency Tradeoffs

- Larger batches usually increase throughput but can raise per-request latency.
- Higher concurrency can improve utilization until memory or scheduling contention appears.
- Stable request/response logging is required to interpret performance changes.

## Output And Logging Design

This lesson requires:

- stable `request_id`
- status for each request (`ok` or `error`)
- response payload and processing timestamps
- run metadata with model, batch, concurrency, and GPU visibility

## Common Failure Modes

See [troubleshooting/common-failures.md](troubleshooting/common-failures.md).

## Operational Checklist

- request schema validated
- IDs preserved end-to-end
- container and GPU visibility confirmed
- batch/concurrency settings documented
- output counts checked
- metrics and summary report saved

## Next Lesson

Suggested next step: reference architectures for customer AI systems on LUMI AI Factory.

