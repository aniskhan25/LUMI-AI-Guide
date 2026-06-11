# 12. Cost Awareness and Capacity Planning

## Goal

Choose, stage, and scale AI workloads on LUMI so GPU-hours are spent deliberately rather than reactively.

By the end of this lesson, you should be able to:

- explain when a workload needs explicit cost and capacity planning
- build a staged run ladder from debug to scaled execution
- define scale-up and stop rules before spending more GPU-hours
- decide what should be reused versus recomputed
- produce a planning brief that another person can review and follow

The practical question in this lesson is:

How should I choose, stage, and scale AI workloads on LUMI so GPU-hours are spent deliberately rather than reactively?

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You completed [06. Topology-Aware Scaling](../06-topology-aware-scaling/README.md), [10. MLOps and Lifecycle Management](../10-mlops-and-lifecycle-management/README.md), and [11. Team Operating Models and Collaboration](../11-team-operating-models-and-collaboration/README.md).
- You already have one existing workflow to plan.

## Working directory

Use this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/12-cost-awareness-and-capacity-planning
```

## What the core concepts mean here

- workload profile:
  what the run is trying to do and what cost drivers matter
- staged run ladder:
  progressively larger runs used to reduce uncertainty before scaling up
- capacity planning:
  deciding when more resources are justified
- artifact reuse:
  reusing expensive intermediate outputs instead of paying to recompute them

This lesson is about better operational decisions, not exact billing prediction.

## When this lesson is needed

Use this lesson when:

- runs are consuming meaningful GPU-hours
- scale decisions are unclear
- the team is repeatedly jumping to large runs too early
- reruns are expensive because outputs are hard to reuse
- queue and partition choices are affecting turnaround

Typical warning signs are:

- multi-node runs happen before a one-device baseline exists
- the same heavy preprocessing runs again and again
- partitions are chosen by habit rather than by stage purpose
- scale-up happens without a written rule for when the next rung is allowed

## What this lesson is and is not

This lesson is:

- a workload-planning tutorial
- a way to reduce waste from premature scaling or recomputation
- a way to tie run purpose to partition and resource choice

This lesson is not:

- financial accounting
- exact billing prediction
- a procurement or quota-management guide
- a substitute for workflow validation or scaling evidence

## The operational transition this lesson teaches

The transition is:

- “submit what seems large enough and see what happens”

to:

- “use a staged ladder and explicit decision gates before spending more GPU-hours”

That transition requires:

- one baseline rung
- one clear question per rung
- one rule for when the next rung is allowed
- one plan for artifact reuse

## Main planning levers

The choices that matter most in this lesson are:

- workload objective:
  debug, baseline measurement, throughput check, or steady-state operation
- run ladder:
  the order of debug, baseline, single-node, and multi-node stages
- partition choice:
  which queue fits the purpose of the rung
- scale choice:
  device count and node count
- turnaround sensitivity:
  whether the run needs quick feedback or production-style throughput
- artifact reuse:
  which outputs should be preserved so you do not pay to recreate them

These levers matter more than a single “best size.”

## How to plan a workload deliberately

Use this order:

1. Start from the smallest run that can answer the current question.
2. Prove correctness before scaling.
3. Separate throughput goals from exploratory runs.
4. Match partition choice to stage purpose.
5. Decide which artifacts should be reused before launching the next rung.
6. Promote scale only when the previous rung justified it.

Use this lesson rule:

Do not move to the next rung unless the current rung answered the question it was meant to answer.

## How to spot bad capacity planning

Warning signs include:

- skipping debug runs and going straight to multi-node
- rerunning expensive preprocessing every time
- choosing partitions by habit instead of purpose
- scaling before evaluation or scaling evidence exists
- keeping no written rule for when the next rung is allowed

Bad planning usually wastes compute through ambiguity rather than through one obviously wrong job.

## Minimal workflow

This lesson is short on commands because the main work is planning.

### Step 1: Study the worked example

Read:

- [Baseline workload](./worked-example/baseline-workload.md)
- [Scaling decision](./worked-example/scaling-decision.md)
- [Production plan](./worked-example/production-plan.md)

These show:

- the baseline question
- the rung-to-rung scale decision
- the eventual steady-state operating choice

### Step 2: Fill the planning templates

Use:

- [Workload profile](./templates/workload-profile.md)
- [Staged run plan](./templates/staged-run-plan.md)
- [Estimate table](./templates/estimate-table.md)
- [Post-run review](./templates/post-run-review.md)

The workload profile should make explicit:

- what the run is trying to achieve
- what the baseline and target configurations are
- what scale-up gate must be passed
- what stop criteria apply

The staged run plan should make explicit:

- the purpose of each rung
- the partition
- the resources
- the walltime
- the decision gate for moving on

### Step 3: Validate the planning brief

Command:

```bash
python assets/validate_capacity_plan.py \
  --profile templates/workload-profile.md \
  --run-plan templates/staged-run-plan.md
```

Expected result:

- `VALIDATION_OK=1`
- the workload profile and staged run plan contain the required sections

This is structural success.

It means the planning brief is complete enough to review.

It does not mean the estimates are wise.

## How to read a good capacity plan

A good capacity plan should answer these questions quickly:

- what question is each rung supposed to answer?
- why is this partition being used?
- what resource request is justified at this stage?
- what metric allows the next rung?
- what artifacts will be reused instead of recomputed?
- what condition tells the team to stop scaling?

If those answers are not obvious, the plan is still too reactive.

## Scale-up rule

The scale-up rule is the core of the lesson:

- Stage 0:
  prove command, path, and environment correctness
- Stage 1:
  establish one-device baseline behavior
- Stage 2:
  test whether larger scale materially improves throughput or turnaround
- Stage 3:
  use multi-node only if the earlier stages justified it

If a rung fails to answer its intended question, fix that before climbing.

## Partition and reuse note

For practical operational reminders, keep these two notes:

- [GPU-hour planning note](./assets/gpu-hour-planning-note.md)
- [Should this be multi-node?](./assets/should-this-be-multi-node.md)

Those are worth keeping separate because they capture two recurring questions:

- is the run decision-worthy at this size?
- is multi-node actually justified?

## What this lesson outcome demonstrates

If this lesson works well, you have shown that:

- a workload can be planned as a staged ladder instead of a guess
- partition choice can be tied to stage purpose
- scale-up can be justified with written gates
- artifact reuse can reduce recomputation waste

That is different from proving the final plan is optimal.

## What to change next

After the first capacity plan, change one thing at a time.

Recommended order:

1. fix the baseline rung before changing partition or scale assumptions
2. improve artifact reuse before increasing compute spend
3. tighten scale-up gates before authorizing multi-node
4. strengthen post-run review before normalizing larger runs

## Cross-lesson map

Lesson 06 taught how to interpret scaling.

Lesson 10 taught how to manage reusable artifacts.

Lesson 11 taught how teams share ownership.

Lesson 12 now asks:

- when is a larger run actually justified?
- what is the cheapest rung that can answer the current question?

## Next lesson

Next extension lesson: domain accelerator modules for priority customer scenarios.
