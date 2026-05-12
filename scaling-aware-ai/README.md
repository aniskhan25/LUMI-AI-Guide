# Scaling-Aware AI on LUMI

Practical guidance for using LUMI-G efficiently for AI workloads.

This standalone guide is for users who already know how to submit jobs on LUMI and now need to decide how far, how fast, and how safely to scale their workloads. The focus is not only on launching distributed jobs. The focus is on making evidence-based scaling decisions from baseline measurements, placement metadata, throughput, efficiency, profiling, and GPU-hour cost.

## Goal

Help LUMI users exploit compute resources effectively by teaching a repeatable scaling workflow:

1. Build a trustworthy single-GCD baseline.
2. Validate placement and launch behavior before interpreting performance.
3. Scale through a deliberate ladder: 1 GCD, 8 GCDs, then multi-node.
4. Separate compute, communication, data, memory, and launch bottlenecks.
5. Decide whether larger runs are worth the added complexity and GPU-hours.

## Audience

This guide is intended for:

- researchers moving from single-device experiments to LUMI-G
- AI engineers tuning distributed PyTorch, DeepSpeed, FSDP, Megatron-style, inference, or embedding workloads
- project teams trying to reduce wasted GPU-hours
- support staff who need a structured way to diagnose poor scaling reports

It assumes basic familiarity with:

- Linux shell usage
- Slurm job submission
- Python AI workflows
- containers on LUMI
- basic GPU training or inference concepts

## What This Guide Will Cover

- LUMI-G topology as seen by AI frameworks
- strong and weak scaling
- scaling metrics: throughput, speedup, efficiency, latency, GPU-hour cost
- single-GCD baselines
- single-node 8-GCD launches
- multi-node launches
- placement, rank mapping, CPU binding, and GPU visibility
- data pipeline scaling
- communication bottlenecks
- profiling and observability
- workload-specific scaling recommendations
- troubleshooting and run validation
- capacity-aware scaling decisions

## What This Guide Is Not

This is not a generic distributed training tutorial, a benchmark leaderboard, or a replacement for official LUMI documentation. It is a practical decision guide for making scaling work well on LUMI.

## Status

Pass 3 is in place. The guide now has:

- a defined product scope
- a target audience
- a standalone repository-style structure
- a chapter architecture
- a technical scope
- a roadmap for runnable examples and scripts
- initial guide chapters
- synthetic scaling configs
- 1-GCD, 8-GCD, and 16-GCD job scripts
- placement, environment, metrics, comparison, and validation scripts
- workload-specific DDP training example
- workload-specific batch inference job-array example

The guide now includes the synthetic ladder, a synchronized training pattern, and an independent batch-processing pattern.

## Repository Layout

```text
scaling-aware-ai/
  README.md
  ROADMAP.md
  CONTRIBUTING.md
  configs/
    batch-inference/
    ddp-training/
    synthetic/
  docs/
    product-brief.md
    content-architecture.md
    technical-scope.md
  examples/
  guide/
  jobs/
  scripts/
  templates/
```

## Recommended Reading Order

1. [Product brief](docs/product-brief.md)
2. [Content architecture](docs/content-architecture.md)
3. [Technical scope](docs/technical-scope.md)
4. [Introduction](guide/01-introduction.md)
5. [LUMI-G mental model](guide/02-lumi-g-mental-model.md)
6. [Scaling metrics](guide/03-scaling-metrics.md)
7. [Synthetic scaling ladder](guide/04-synthetic-scaling-ladder.md)
8. [Workload taxonomy](guide/05-workload-taxonomy.md)
9. [Data pipeline scaling](guide/06-data-pipeline-scaling.md)
10. [Workload-specific examples](guide/07-workload-specific-examples.md)
11. [Roadmap](ROADMAP.md)
12. [Contributing guide](CONTRIBUTING.md)

## Quick Start: Synthetic Scaling Ladder

From this directory, edit the `#SBATCH --account=project_XXXXXXXXX` lines in `jobs/`, then submit:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
sbatch jobs/run_16gcd_two_node.sh
```

After all three jobs finish:

```bash
python scripts/compare_scaling.py
python scripts/validate_scaling_run.py
```

The main outputs are written under `outputs/`.

## Workload Examples

DDP training:

```bash
sbatch jobs/run_ddp_1gcd.sh
sbatch jobs/run_ddp_8gcd_single_node.sh
```

Batch inference job array:

```bash
sbatch jobs/run_batch_inference_array.sh
python scripts/collect_batch_inference.py --config configs/batch-inference/job_array.yaml
```

## Core Principle

Do not scale an unstable or undersized workload.

Scaling on LUMI should be treated as a controlled experiment. A larger run is only useful if it improves useful throughput enough to justify the communication, launch complexity, queue time, and GPU-hour cost.
