# 4. Synthetic Scaling Ladder

This chapter runs the first complete scaling-aware workflow in the guide.

The ladder compares:

- 1 GCD
- 8 GCDs on one LUMI-G node
- 16 GCDs across two LUMI-G nodes

The workload performs repeated dense GPU compute and a distributed all-reduce when world size is greater than 1. It is designed to expose launch, placement, and communication behavior. It is not intended to represent every AI model.

## Working Directory

Run commands from:

```bash
cd /path/to/scaling-aware-ai
```

If you are using this guide inside the current repository:

```bash
cd /path/to/LUMI-AI-Guide/scaling-aware-ai
```

## Prerequisites

Set `CONTAINER` to a valid LUMI AI container. If this guide is inside `LUMI-AI-Guide`, the job scripts will source `../env.sh` if it exists.

Before submitting, edit the `#SBATCH --account=project_XXXXXXXXX` line in each job script.

## Scripts

The ladder uses:

- `jobs/run_1gcd.sh`
- `jobs/run_8gcd_single_node.sh`
- `jobs/run_16gcd_two_node.sh`
- `scripts/summarize_environment.py`
- `scripts/inspect_placement.py`
- `scripts/run_synthetic_workload.py`
- `scripts/collect_metrics.py`
- `scripts/compare_scaling.py`
- `scripts/validate_scaling_run.py`

## Step 1: Submit The Ladder

Submit the jobs:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
sbatch jobs/run_16gcd_two_node.sh
```

The jobs can run independently. Build the comparison only after all three complete.

## Step 2: Build The Comparison Report

```bash
python scripts/compare_scaling.py
```

Expected report files:

```text
outputs/scaling_report.json
outputs/scaling_report.md
```

## Step 3: Validate The Run

```bash
python scripts/validate_scaling_run.py
```

Expected success marker:

```text
VALIDATION_OK=1
```

Validation checks that:

- every run has a summary
- every run has an environment summary
- metrics files exist
- placement files exist
- observed world size matches the config
- observed node count matches the config
- comparison reports exist

## Output Layout

The ladder writes:

```text
outputs/
  synthetic-1gcd/
    environment.json
    raw/
      placement_rank0.json
      metrics_rank0.json
    run_summary.json
  synthetic-8gcd-single-node/
    environment.json
    raw/
      placement_rank0.json
      metrics_rank0.json
      ...
    run_summary.json
  synthetic-16gcd-two-node/
    environment.json
    raw/
      placement_rank0.json
      metrics_rank0.json
      ...
    run_summary.json
  scaling_report.json
  scaling_report.md
```

## How To Interpret The Result

Start with `outputs/scaling_report.md`.

Good scaling looks like:

- 8 GCDs materially outperform 1 GCD
- 16 GCDs materially outperform 8 GCDs
- efficiency drops, but not catastrophically
- placement validation passes

Weak single-node scaling suggests:

- workload too small
- rank placement issue
- data or CPU-side bottleneck in a real workload
- synchronization overhead

Good single-node scaling but weak two-node scaling suggests:

- inter-node communication is limiting
- the workload is not large enough for multi-node
- network or rendezvous settings should be inspected

Invalid scaling means:

- wrong world size
- wrong node count
- missing placement files
- missing metrics
- inconsistent rank output

Do not interpret invalid runs as performance results.

## What To Change First

If the result is poor but valid, change one thing at a time:

1. Increase `samples_per_step`.
2. Increase `hidden_size` or `compute_repeats`.
3. Compare one-node behavior again.
4. Only then revisit multi-node settings.

## Practical Rule

The first successful ladder does not prove that your real model will scale. It proves that your launch, measurement, and interpretation workflow is ready.

