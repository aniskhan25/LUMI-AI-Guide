# Architecture D: High-Throughput Inference Factory

## Best For

- Large-scale embedding, generation, scoring, and transformation workloads over big corpora or queues.

## Core Flow

request/corpus queue -> batched inference on LUMI-G -> structured outputs -> downstream handoff

## Required Components

- Stable request schema
- Batched/service-style inference runner
- Response/error logging with IDs
- Throughput/latency summary and operating report

## Compute Placement

- LUMI-G: high-throughput model execution
- orchestration and downstream post-processing can run on lighter environments

## Data Pattern

- request/response artifacts versioned
- optional output sharing/staging through LUMI-O
- optional curated input source through Dataset-as-a-Service

## Evaluation Gate

Minimum gate before wider pilot:

- completion rate and error rate thresholds met
- throughput target met for planned workload volume
- response schema validated end-to-end

## First Risk To Watch

- Building unnecessary serving complexity when a disciplined batch factory pattern is sufficient.

## Operational Checklist

- request IDs preserved end-to-end
- batching/concurrency settings documented
- throughput and latency measured consistently
- downstream handoff contract validated

