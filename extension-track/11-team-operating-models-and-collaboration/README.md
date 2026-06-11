# 11. Team Operating Models and Collaboration

## Goal

Define how a team divides ownership, review, sharing, and handoff so an AI workflow remains understandable and operable as more people contribute.

By the end of this lesson, you should be able to:

- explain when a workflow needs explicit team operating rules
- assign artifact and decision ownership clearly
- define review and promotion boundaries
- make dataset and artifact source-of-truth rules explicit
- produce a handoff package that another contributor or team can use without ambiguity

The practical question in this lesson is:

How should a team divide ownership, review, and handoff so an AI workflow remains understandable and operable as more people contribute?

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You completed [10. MLOps and Lifecycle Management](../10-mlops-and-lifecycle-management/README.md).
- You already have one workflow that now involves more than one contributor.

## Working directory

Use this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/11-team-operating-models-and-collaboration
```

## What the core concepts mean here

- role ownership:
  who is responsible for a specific artifact or decision
- review gate:
  who must sign off before promotion or wider use
- handoff:
  the minimum package another role needs to continue safely
- source of truth:
  the one authoritative location or artifact for a given class of information

This lesson is about explicit operating boundaries, not broad project management.

## When this lesson is needed

Use this lesson when:

- multiple contributors now touch the same workflow
- datasets, evaluations, and promoted artifacts are shared
- handoffs are causing confusion or duplicated work
- promotion requires more than one person’s judgment
- outputs are being delivered to another team

Typical warning signs are:

- two people both think they own the same artifact
- handoff happens through chat without stable metadata
- more than one dataset copy is treated as authoritative
- promoted artifacts exist without clear approver context

## What this lesson is and is not

This lesson is:

- a practical team operating-model tutorial
- a way to make ownership and review explicit
- a way to make handoff and sharing repeatable

This lesson is not:

- a full org chart
- HR process
- enterprise program management
- a substitute for lifecycle discipline or evaluation evidence

## The operational transition this lesson teaches

The transition is:

- “one person can keep the workflow in their head”

to:

- “the workflow now depends on explicit ownership, review, and handoff boundaries”

Once that transition happens, technical success alone is not enough.

The workflow also needs:

- one owner per artifact class
- explicit approvers for promotion
- one source of truth per dataset version
- a handoff package another person can actually use

## Main operating levers

The choices that matter most in this lesson are:

- who owns datasets
- who owns workflow runs and configs
- who owns evaluation and promotion review
- who publishes delivery-ready artifacts
- which path is the source of truth for each artifact class
- what minimum package must be present at handoff

These levers matter more than job titles.

## How to divide ownership well

Use this order:

1. Assign ownership by artifact or decision, not vague role labels.
2. Keep one accountable owner per artifact class.
3. Separate artifact production from promotion approval where needed.
4. Keep the handoff package small but sufficient.
5. Make review gates visible and repeatable.

Use this lesson rule:

An artifact is not team-ready unless its owner, reviewer, intended use, and handoff path are all explicit.

## How to spot a weak team model

Warning signs include:

- two contributors both believe they own the same artifact
- nobody owns evaluation or promotion decisions
- handoffs happen through chat only
- there are multiple “source of truth” paths
- a downstream team receives outputs without limitations or reviewer context
- the workflow stalls when one person is unavailable

Weak collaboration usually shows up as ambiguity, not immediate technical failure.

## Minimal workflow

This lesson is short on commands because the work is operational.

### Step 1: Study the worked example

Read:

- [Project scenario](./worked-example/project-scenario.md)
- [Collaboration pattern](./worked-example/collaboration-pattern.md)
- [Sharing model](./worked-example/sharing-model.md)

These show:

- the project context
- the collaboration risk
- the chosen team model
- the sharing boundary between internal and cross-project use

### Step 2: Fill the team operating artifacts

Use:

- [Team operating model](./templates/team-operating-model.md)
- [Responsibility matrix](./templates/responsibility-matrix.md)
- [Artifact flow map](./templates/artifact-flow-map.md)
- [Handoff checklist](./templates/handoff-checklist.md)

The operating model should make explicit:

- team roles
- artifact owners
- source-of-truth paths
- storage and sharing boundaries
- review and promotion rules
- handoff contract

The responsibility matrix should make ambiguous ownership visible before it causes operational drift.

### Step 3: Validate the blueprint structure

Command:

```bash
python assets/validate_collaboration_blueprint.py \
  --blueprint templates/team-operating-model.md \
  --matrix templates/responsibility-matrix.md
```

Expected result:

- `VALIDATION_OK=1`
- blueprint and responsibility matrix contain the required sections

This is structural success.

It means the collaboration blueprint is complete enough to review.

It does not mean the team will automatically collaborate well.

## How to read a good operating model

A good operating model should answer these questions quickly:

- who owns each artifact class?
- who reviews promotion decisions?
- which dataset path is authoritative?
- what is shared internally versus across projects?
- what must be included in the handoff package?
- who is the escalation contact if something breaks?

If those answers are not obvious, the workflow is still too dependent on informal memory.

## Review and promotion rule

A promoted artifact should not be handed off unless:

- the owner is explicit
- the approver is explicit
- the source run and evaluation context are present
- intended use and known limitations are included
- the retrieval or storage path is unambiguous

This is the human operating layer on top of Lesson 10 lifecycle discipline.

## Dataset and sharing note

For the specific question of curated upstream datasets versus self-managed working copies, use:

- [DaaS vs self-managed dataset note](./assets/daas-vs-self-managed-note.md)

That note is worth keeping separate because it captures a recurring collaboration boundary:

- DaaS can be the authoritative upstream input
- project working copies can still exist for experiments
- the team must keep those roles explicit

## What this lesson outcome demonstrates

If this lesson works well, you have shown that:

- the workflow can be operated by more than one person without relying on memory
- artifact ownership and review boundaries are explicit
- dataset source-of-truth rules are visible
- handoff can happen with less ambiguity

That is different from proving the team structure is ideal or permanent.

## What to change next

After the first operating model, change one thing at a time.

Recommended order:

1. resolve duplicate ownership before expanding sharing
2. make the source-of-truth dataset explicit before adding more working copies
3. strengthen review gates before increasing promotion volume
4. tighten handoff requirements before onboarding downstream teams

## Cross-lesson map

Lesson 10 made workflows reusable assets.

Lesson 11 assigns the human operating model around those assets:

- who owns them
- who reviews them
- who hands them off

Lesson 12 then builds on this by asking how those workflows should be planned and staged against cost and capacity constraints.

## Next lesson

Next extension lesson: cost awareness, capacity planning, and workload selection on LUMI-G.
