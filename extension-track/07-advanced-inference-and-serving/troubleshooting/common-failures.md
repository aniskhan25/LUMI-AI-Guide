# Common Failures

## 1) Model runs on CPU unexpectedly

Symptoms:

- slow throughput
- metadata shows `gpu_visible_count=0`

Fix:

- verify GPU partition and container bindings
- enforce `runtime.require_gpu=true` for LUMI runs

## 2) Request/response count drift

Symptoms:

- missing responses for some request IDs

Fix:

- ensure every failed request writes an error record
- validate ID join after each run

## 3) Batch size too high

Symptoms:

- instability or out-of-memory errors

Fix:

- reduce batch size and rerun
- compare throughput/latency, not just success/failure

## 4) Concurrency harms stability

Symptoms:

- higher error rate at larger concurrency

Fix:

- tune concurrency incrementally
- keep controlled comparisons with fixed request set

## 5) Inconsistent measurement windows

Symptoms:

- misleading throughput comparisons

Fix:

- use same request set and same metric script
- compare summaries produced by `summarize_results.py`

