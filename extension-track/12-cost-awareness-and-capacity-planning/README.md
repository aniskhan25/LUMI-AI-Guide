# 12. Cost Awareness, Capacity Planning, and Workload Selection on LUMI-G

This lesson teaches how to choose, size, and stage AI workloads on LUMI-G so GPU-hours are used deliberately.

## What This Lesson Enables

Build a practical workload planning brief with:

- workload profile and objective
- staged run ladder from debug to scaled execution
- estimate table and scale-up decision gate
- partition selection rationale
- artifact reuse and recomputation controls

## When This Workflow Is Needed

Use this lesson when:

- runs are consuming meaningful GPU-hours
- scale decisions are unclear
- pilots are moving toward regular operation
- teams need repeatable capacity planning

## What You Need Before Starting

- completion of onboarding guide
- preferred completion of Lessons 6, 7, and 10
- one existing workflow to plan
- ability to inspect usage with `lumi-allocations`

## Workflow At A Glance

```mermaid
flowchart LR
  A["Workload goal"] --> B["Small test run"]
  B --> C["Baseline measurement"]
  C --> D["Scale decision gate"]
  D --> E["Partition and runtime choice"]
  E --> F["Production pattern"]
  F --> G["After-action review"]
```

## Minimal Worked Example

This lesson includes:

- [baseline workload](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/worked-example/baseline-workload.md)
- [scaling decision](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/worked-example/scaling-decision.md)
- [production plan](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/worked-example/production-plan.md)

Core templates:

- [workload profile](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/templates/workload-profile.md)
- [staged run plan](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/templates/staged-run-plan.md)
- [estimate table](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/templates/estimate-table.md)
- [post-run review](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/templates/post-run-review.md)

Supporting assets:

- [partition cheat sheet](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/assets/partition-cheatsheet.md)
- [GPU-hour planning note](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/assets/gpu-hour-planning-note.md)
- [multi-node worksheet](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/assets/should-this-be-multi-node.md)

## How To Verify It Worked

A valid planning brief should show:

- defined baseline stage
- explicit debug/test/scaled stages
- intentional partition choice
- scale-up decision based on measured result
- actual-vs-expected review fields

Optional validation:

```bash
python assets/validate_capacity_plan.py \
  --profile templates/workload-profile.md \
  --run-plan templates/staged-run-plan.md
```

## Billing And Resource Units That Matter

Use these units in every plan:

- GPU-hours for GPU compute
- CPU-core-hours for CPU compute
- TB-hours for storage
- project usage view via `lumi-allocations`

## Choosing The Right Scale

Use a staged ladder:

- 1 GCD for smoke tests and small baselines
- 1 node for serious throughput checks
- multi-node only when measured need justifies it
- `dev-g` for debugging and quick tests, not steady operation

## Partition And Runtime Planning

- choose partition based on intent, not convenience
- set walltime to realistic bounds
- avoid launching maximum scale before baseline data exists

## Storage And Recomputation Choices

- preserve expensive intermediate artifacts
- separate reusable artifacts from throwaway outputs
- prefer container-pinned execution to reduce environment drift
- avoid unnecessary reruns caused by poor artifact discipline

## LUMI Planning Notes

- LUMI-G node appears as 8 GPU-visible devices (MI250X GCD view)
- `dev-g` has strict limits and is for quick debug use
- storage and compute are billed units, so staged planning matters
- container-first workflows reduce repeated setup overhead

## Common Failure Modes

See [common-failures.md](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning/troubleshooting/common-failures.md).

## Operational Checklist

- workload objective defined
- baseline measured
- partition selected for explicit reason
- scale-up and stop rules documented
- artifact reuse strategy documented
- post-run review completed

## Next Lesson

Suggested next step: domain accelerator modules for priority customer scenarios.
