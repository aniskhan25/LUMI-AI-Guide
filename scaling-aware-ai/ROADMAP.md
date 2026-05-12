# Roadmap

## Pass 1: Product Definition

Status: complete.

Deliverables:

- standalone guide directory
- product brief
- audience definition
- content architecture
- technical scope
- roadmap
- contribution guidance

## Pass 2: First Runnable Vertical Slice

Status: complete.

Goal:

Create the first useful, runnable version of the guide.

Deliverables:

- `guide/01-introduction.md`
- `guide/02-lumi-g-mental-model.md`
- `guide/03-scaling-metrics.md`
- `guide/04-synthetic-scaling-ladder.md`
- `jobs/run_1gcd.sh`
- `jobs/run_8gcd_single_node.sh`
- `jobs/run_16gcd_two_node.sh`
- `scripts/inspect_placement.py`
- `scripts/run_synthetic_workload.py`
- `scripts/collect_metrics.py`
- `scripts/compare_scaling.py`
- `scripts/validate_scaling_run.py`
- `configs/synthetic/*.yaml`
- `templates/scaling-report.md`

Acceptance criteria:

- a user can run a 1-GCD, 8-GCD, and 16-GCD synthetic ladder
- every run emits placement and metrics artifacts
- the comparison report includes throughput, speedup, and efficiency
- validation detects missing outputs or mismatched world sizes
- the guide explains how to interpret good, poor, and invalid results

## Pass 3: Workload-Specific Expansion

Status: complete.

Goal:

Move beyond synthetic scaling into representative AI workloads.

Candidate additions:

- DDP training example
- synthetic data versus real data comparison
- batch inference or embedding example
- job-array pattern for embarrassingly parallel AI workloads
- checkpoint timing example
- data format and input pipeline discussion

Acceptance criteria:

- users can distinguish distributed training problems from data pipeline problems
- examples include workload-specific metrics
- recommendations identify when distributed scaling is unnecessary

## Pass 4: Profiling and Diagnosis

Goal:

Add deeper debugging guidance for poor scaling.

Candidate additions:

- lightweight metrics checklist
- PyTorch profiler walkthrough
- ROCm profiler entry points
- per-rank variance analysis
- communication-heavy versus compute-heavy workload comparison
- troubleshooting catalog

Acceptance criteria:

- users can identify likely bottleneck category before changing launch scripts
- profiling guidance is staged from low overhead to high detail

## Pass 5: Production Planning

Goal:

Connect scaling decisions to project-level resource planning.

Candidate additions:

- capacity planning template
- GPU-hour cost examples
- stop/go scale-up gates
- artifact reuse strategy
- post-run review template
- production run checklist

Acceptance criteria:

- users can document why a given scale was selected
- users can estimate and review GPU-hour use
- production plans include rollback and restart assumptions

## Deferred Ideas

These are useful but should not block the early guide:

- DeepSpeed-specific chapter
- FSDP-specific chapter
- Megatron-style tensor/pipeline parallel chapter
- vLLM or service-oriented inference chapter
- automated plotting of scaling reports
- CI-style validation for generated example outputs
- HTML documentation site
