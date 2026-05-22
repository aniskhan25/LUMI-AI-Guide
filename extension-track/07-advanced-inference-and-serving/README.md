# 07. Advanced Inference and Serving

## Goal

Compare two repeated-inference operating patterns on LUMI and decide whether a queued batched path or a service-style loop is the better fit.

By the end of this lesson, you should be able to:

- explain the difference between batched inference and a service-style loop
- run one controlled comparison of both patterns on LUMI
- validate that requests, responses, errors, and metrics stay aligned
- justify which pattern better fits the request pattern you tested

The practical question in this lesson is:

When should I use batched inference, and when should I keep a service-style loop inside a scheduled LUMI job?

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You already know how to run Python and submit a batch job on LUMI.
- `../../env.sh` is configured with a valid `CONTAINER`.

## Working directory

Run commands in this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/07-advanced-inference-and-serving
```

## What the two patterns mean here

- batched inference:
  process a queued request set in larger groups to maximize throughput
- service-style loop:
  keep one allocation alive and process repeated internal requests in smaller groups for lower turnaround inside that allocation

This lesson does not cover:

- public internet-facing APIs
- gateway or auth architecture
- autoscaling platforms
- always-on production serving stacks

This is a scheduled-job lesson, not a cloud serving lesson.

## Why this lesson uses a synthetic model path

The code uses a synthetic/template inference function on purpose.

That choice keeps the lesson focused on:

- request accounting
- batch size and concurrency behavior
- latency and throughput measurement
- output integrity
- operating-pattern comparison

It does not claim to benchmark a real inference stack such as vLLM, TGI, or a production endpoint.

Use this lesson as an operating-pattern tutorial, not a serving benchmark.

## When to use each pattern

Use batched inference when:

- requests are already queued
- throughput matters more than per-request turnaround
- you want a simple, repeatable offline path

Use a service-style loop when:

- repeated internal requests arrive during one allocation window
- lower turnaround matters more than peak throughput
- you still want to stay inside a scheduled HPC job model

Use a different architecture entirely when:

- the endpoint must always be available
- public access, authentication, or gateway concerns dominate
- you need autoscaling or broader service lifecycle management

## Main quality levers

The main choices that control behavior in this lesson are:

- batch size:
  larger batches usually improve throughput until memory or queueing costs dominate
- concurrency:
  more concurrent work can improve utilization, but can also raise contention and instability
- request size:
  longer prompts or larger token budgets change both latency and throughput
- completion rate:
  faster throughput is not useful if more requests fail
- latency target:
  p95 latency is often the more useful operating number than average latency

These are the levers to reason about before changing model code or hardware.

## What the scripts produce

Each run writes:

- `requests.jsonl`
- `responses.jsonl`
- `errors.jsonl`
- `run_metadata.json`
- `metrics.json`
- `summary.json`

The main run directories are:

- `outputs/advanced-inference-batched`
- `outputs/advanced-inference-service`

The comparison step writes:

- `outputs/advanced-inference-comparison.json`
- `outputs/advanced-inference-comparison.md`

The critical integrity rule is simple:

Every request must appear exactly once as either a response or an error.

## Minimal workflow

Load the lesson runtime in your shell:

```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch
source ../../env.sh
```

### Step 1: Submit the two runs

Commands:

```bash
sbatch jobs/run_batched_inference.sh
sbatch jobs/run_service_style_inference.sh
```

What they do:

- `run_batched_inference.sh`:
  runs the queued batched path and writes metrics for the batched configuration
- `run_service_style_inference.sh`:
  runs the service-style loop and writes metrics for the service configuration

### Step 2: Build the comparison

Command:

```bash
python scripts/summarize_results.py
```

This compares:

- throughput
- p95 latency
- completion rate
- error rate

and writes a simple recommendation.

### Step 3: Validate outputs

Command:

```bash
python scripts/validate_inference_run.py
```

Expected result:

- both run directories exist
- requests, responses, errors, metadata, metrics, and summaries exist
- request coverage is complete in both runs
- `advanced-inference-comparison.json` and `.md` exist
- `VALIDATION_OK=1`

This is structural success. It means the two operating patterns were run and compared correctly.

It does not yet mean one pattern is universally better.

## How to read the result

Start with:

- `outputs/advanced-inference-batched/summary.json`
- `outputs/advanced-inference-service/summary.json`
- `outputs/advanced-inference-comparison.md`

The main numbers are:

- `throughput_rps`
- `p95_latency_ms`
- `completion_rate`
- `error_rate`

A stronger batched result looks like:

- clearly higher throughput
- acceptable latency for the use case
- no completion-rate regression

A stronger service-style result looks like:

- materially lower p95 latency or turnaround
- stable completion rate
- acceptable throughput for the request arrival pattern

Use this lesson rule:

Do not choose a pattern on throughput alone if it causes a meaningful latency or completion-rate regression.

## How to diagnose a weak result

When the comparison looks weak or inconclusive, ask:

1. Is the request set large enough to produce a meaningful comparison?
2. Is the real issue batch size or concurrency rather than the operating pattern itself?
3. Did error rate rise as concurrency increased?
4. Is GPU visibility missing, making the comparison invalid?
5. Is the workflow really a serving problem, or just an offline scheduling problem?

Interpretation:

- poor batched throughput:
  inspect batch size before redesigning the pattern
- poor service-style stability:
  inspect concurrency before concluding the pattern is wrong
- equal metrics on both paths:
  the request pattern may be too small or too simple to distinguish them

## What this lesson does and does not prove

If the lesson works end to end, it shows that:

- request IDs can be tracked end to end
- metrics and integrity checks can be compared across two operating patterns
- a queued batch path and a service-style loop can be evaluated on the same request set
- the pattern choice can be justified with saved artifacts

It does not prove:

- that this synthetic path predicts a real production inference engine
- that one pattern will always win on every workload
- that a service-style loop on LUMI replaces a public serving platform

## What to change next

After the first successful run, change one thing at a time.

Recommended order:

1. Increase request-set size before changing architecture conclusions.
2. Tune batch size before tuning concurrency.
3. Tune concurrency only after output integrity and error rate stay stable.
4. Move to a real inference backend only after the operating-pattern comparison is understood.

## Troubleshooting

- `gpu_visible_count=0`:
  fix the launch or container binding before trusting the numbers
- missing request coverage:
  inspect `errors.jsonl` and ID handling before interpreting performance
- higher throughput with much worse p95 latency:
  treat it as a tradeoff, not an automatic win
- higher concurrency with higher error rate:
  back off concurrency before changing the serving pattern

## Next lesson

Next extension lesson: reference architectures for customer AI systems on LUMI AI Factory.
