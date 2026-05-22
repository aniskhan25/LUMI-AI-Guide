# 10. MLOps and Lifecycle Management

## Goal

Turn a successful workflow from a one-off experiment into a reusable team asset with explicit provenance, evaluation linkage, promotion state, and handoff information.

By the end of this lesson, you should be able to:

- explain when a workflow needs lifecycle discipline rather than more experimentation
- record the minimum metadata needed to reproduce and compare runs
- separate draft experiments from promoted artifacts
- define when an artifact is eligible for promotion
- produce a handoff-ready manifest and summary for another team member

The practical question in this lesson is:

When is a successful workflow no longer just an experiment, and what must change to manage it as a reusable team asset?

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You completed the workflow lessons that produced the artifact you now want to manage.
- You already have one existing workflow artifact set to organize.

## Working directory

Use this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/10-mlops-and-lifecycle-management
```

## What the core concepts mean here

- experiment:
  a run whose main purpose is to learn something
- promoted artifact:
  a run output that is now intentionally reusable
- lifecycle management:
  the rules that connect datasets, configs, models, evaluations, ownership, and storage over time
- handoff:
  the minimum information another person needs to reuse or audit the workflow safely

This lesson is about workflow discipline, not platform automation.

## When this lesson is needed

Use this lesson when:

- multiple people now touch the same workflow
- repeated reruns must stay comparable
- artifacts are accumulating and becoming hard to trace
- a team wants to reuse outputs instead of recomputing them
- promotion decisions need more discipline than “this looks good enough”

Typical warning signs are:

- output folders named `latest`, `final`, or `final-final`
- nobody can tell which dataset version produced a result
- evaluation exists, but is detached from the promoted artifact
- only the original author knows what should actually be reused

## What this lesson is and is not

This lesson is:

- a lightweight lifecycle-management tutorial
- a way to keep workflow artifacts reusable and comparable
- a way to make promotion and handoff explicit

This lesson is not:

- a full enterprise MLOps platform
- CI/CD or registry infrastructure
- a substitute for evaluation or trust review
- a complete governance framework

## The operational transition this lesson teaches

The transition is:

- “I ran something successfully once”

to:

- “we can now reproduce, compare, promote, retire, and hand off this workflow”

That transition requires more than storing files. It requires:

- pinned inputs
- explicit lifecycle state
- evaluation linkage
- promotion rules
- ownership and storage boundaries

## Main lifecycle levers

The choices that matter most in this lesson are:

- what counts as a versioned input:
  dataset, config, model or adapter, and runtime environment
- where evaluation attaches:
  every reusable artifact must link back to evidence
- how lifecycle states are separated:
  draft, reviewed, promoted, retired, archived
- where artifacts live:
  active workspace versus promoted storage versus optional sharing path
- how ownership is recorded:
  who owns the run, who reviewed it, and when
- what minimum handoff data is required:
  enough for someone else to rerun, inspect, or retire the asset

These levers matter more than tool branding.

## How to manage a workflow as an asset

Use this order:

1. Pin the dataset, config, model or adapter, and container reference.
2. Give the run one stable run ID.
3. Attach evaluation to the run, not to memory or chat history.
4. Separate draft experiments from promoted artifacts.
5. Record who owns the artifact and who reviewed it.
6. Record where the promoted copy lives and how another team finds it.

Use this lesson rule:

Do not promote an artifact unless its inputs, evaluation evidence, owner, and storage path are all explicit.

## How to spot weak lifecycle management

Warning signs include:

- the team cannot tell which dataset produced a model or output
- a promoted artifact has no linked evaluation evidence
- experiments and promoted assets are stored together
- reruns are reproducible only by the original author
- storage paths exist, but ownership or intended use is unclear
- recomputation happens because prior outputs are hard to find

Weak lifecycle management usually shows up as ambiguity, not immediate breakage.

## Minimal workflow

This lesson is short on commands because the main work is organizational.

### Step 1: Study the lifecycle example

Read:

- [Baseline experiment](./worked-example/baseline-experiment.md)
- [Promoted version](./worked-example/promoted-version.md)
- [Shareable artifacts plan](./worked-example/shareable-artifacts.md)

These show the transition from:

- inconsistent run artifacts

to:

- a promoted, handoff-ready package

### Step 2: Fill the manifest and lifecycle templates

Use:

- [Run manifest](./templates/run-manifest.yaml)
- [Lifecycle states](./templates/lifecycle-states.md)
- [Promotion checklist](./templates/promotion-checklist.md)
- [Artifact layout](./templates/artifact-layout.md)

The manifest is the central artifact.

It should make explicit:

- the run ID
- lifecycle state
- owner
- dataset, config, model, and container versions
- input/output/evaluation/promoted/share paths
- evaluation summary
- promotion status

### Step 3: Validate the manifest

Command:

```bash
python scripts/validate_manifest.py --manifest templates/run-manifest.yaml
```

Expected result:

- `VALIDATION_OK=1`
- the required lifecycle fields are present

This is structural success.

It means the manifest is complete enough to review.

It does not mean the workflow is operationally ready for promotion.

### Step 4: Build the run summary

Command:

```bash
python scripts/build_run_summary.py --manifest templates/run-manifest.yaml
```

This produces a compact markdown and JSON summary that another person can read without scanning the whole manifest.

## How to read a good lifecycle record

A good lifecycle record should answer these questions quickly:

- what exactly was run?
- which versions of data, config, model, and container were used?
- where are the outputs and evaluation?
- is this still a draft experiment, or is it promoted?
- who owns it?
- what is it intended for, and what should not be assumed?

If those answers are not obvious, the workflow is still too author-dependent.

## Promotion and retirement rule

Promotion should happen only when:

- provenance is complete
- evaluation is attached
- intended use is documented
- known limitations are explicit
- owner and reviewer are identified

Retirement should happen when:

- a promoted artifact is no longer recommended
- a better replacement exists
- follow-up evaluation invalidates the earlier promotion

Retirement is better than leaving ambiguous “final” artifacts in place.

## What this lesson outcome demonstrates

If this lesson works well, you have shown that:

- a workflow can be described as a reusable asset
- provenance and evaluation can stay attached to artifacts
- experiment and promoted states can be separated
- handoff can happen with less ambiguity

That is different from proving the team has a full MLOps platform.

## What to change next

After the first lifecycle record, change one thing at a time.

Recommended order:

1. fix naming and run-state ambiguity before adding more metadata
2. attach evaluation before broadening promotion rules
3. separate draft and promoted storage before sharing widely
4. tighten reviewer and ownership rules before expanding team usage

## Cross-lesson map

Use earlier lessons for the workflow itself:

- Lesson 08:
  choose the architecture
- Lesson 09:
  redesign for sensitive data and stronger trust requirements

This lesson changes the question to:

- how do we keep the workflow reusable, reviewable, and handoff-ready over time?

## Next lesson

Next extension lesson: team operating models and collaboration patterns for AI Factory projects.
