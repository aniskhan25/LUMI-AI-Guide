# 08. Reference Architectures

## Goal

Choose the right end-to-end AI system pattern for a use case on LUMI and justify that choice in a constrained architecture brief.

By the end of this lesson, you should be able to:

- explain what a reference architecture means in this guide
- choose among four recurring AI Factory patterns
- justify why one pattern fits better than the others
- produce a brief that makes scope, placement, evaluation, and pilot boundaries explicit

The practical question in this lesson is:

Which reference architecture should I choose for this AI use case on LUMI, and why?

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You understand the workflow patterns introduced in Lessons 01 through 07.

## Working directory

Use this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/08-reference-architectures
```

## What a reference architecture means here

A reference architecture in this guide is:

- a reusable starting pattern
- a way to reduce design drift across similar customer problems
- a structured way to compare options before building too much

It is not:

- a production blueprint copied without change
- a guarantee that every component belongs on LUMI-G
- a substitute for evaluation, security, or pilot scoping

The point is to start from a small set of proven system shapes instead of designing from scratch every time.

## How this lesson fits the track

Lessons 01 through 07 taught workflow patterns:

- adaptation
- embeddings and inference
- RAG
- evaluation
- synthetic-data loops
- scaling
- repeated inference operating patterns

Lesson 08 changes the question.

It is no longer:

- how do I run this one workflow well?

It becomes:

- which workflow shape should define the system?
- which components are actually necessary in the first pilot?
- what should be deliberately left out?

That is why this lesson is more descriptive than the earlier execution-heavy lessons.

## The four reference architectures

This lesson uses four patterns:

- [Architecture A: Adapt-and-Apply](./architectures/adapt-and-apply.md)
- [Architecture B: Retrieval-Grounded Knowledge Assistant](./architectures/retrieval-grounded-assistant.md)
- [Architecture C: Evaluate-and-Improve Loop](./architectures/evaluate-and-improve-loop.md)
- [Architecture D: High-Throughput Inference Factory](./architectures/high-throughput-inference-factory.md)

They are not interchangeable. Each exists because a different problem is dominant.

## Comparison-first selection guide

Use this matrix before touching the template.

| Pattern | Best when | Why it fits LUMI | Success depends on | Usual failure mode |
|---|---|---|---|---|
| Adapt-and-Apply | model behavior must change for a stable task | LUMI-G is well suited for adaptation and repeated inference | good dataset lineage and evaluation gate | adapting when retrieval would solve the problem faster |
| Retrieval-Grounded Knowledge Assistant | answers must stay grounded in changing documents | embedding and generation-heavy stages fit LUMI-G well | stable chunk/evidence IDs and retrieval quality | blaming the model when retrieval is the real weakness |
| Evaluate-and-Improve Loop | quality is uncertain and stakeholders need before/after proof | repeated generation and scoring runs fit GPU batch execution | fixed benchmark, stable scoring, explicit failure review | scaling complexity before baseline quality is understood |
| High-Throughput Inference Factory | large queued corpus or request processing dominates | batched GPU inference is the main value | stable schema, completion/error accounting, throughput discipline | adding serving or RAG complexity that the task does not need |

## How to choose well

Start with the use case and the constraint that matters most.

Then ask:

1. Is the main problem task behavior, stale knowledge, uncertain quality, or processing volume?
2. What is the smallest pattern that solves that problem?
3. What is the evaluation gate that would prove the pilot is good enough?
4. What components are explicitly out of scope for the first version?

Use this lesson rule:

Choose the simplest architecture that satisfies the real customer constraint.

That usually means:

- do not add adaptation unless retrieval or prompting is insufficient
- do not add RAG unless grounding is actually required
- do not add a synthetic-data loop unless a measured weakness justifies it
- do not add serving complexity if a batch or scheduled internal loop is enough

## Main quality levers

The architecture choices that matter most in this lesson are:

- where task behavior changes:
  prompt only, retrieval, or adaptation
- where grounding is required:
  answer generation with evidence, or pure transformation
- where evaluation enters:
  before pilot, before promotion, or before expansion
- where heavy compute runs:
  LUMI-G only for the stages that truly need GPU acceleration
- where data and artifacts live:
  project storage, LUMI-O staging, or curated datasets
- where human review enters:
  trust gate, failure review, or constrained pilot checkpoint

These levers matter more than component count.

## How to spot a bad architecture choice

Common bad choices look like this:

- adapting a model when the real issue is missing or stale knowledge
- building RAG for a pure throughput transformation task
- adding a serving layer before the batch path is stable
- scaling infrastructure before quality has been evaluated
- forcing every component onto LUMI-G whether it is compute-heavy or not
- mixing too many patterns into the first pilot

A bad architecture is usually not “technically impossible.” It is usually just more complex than the problem requires.

## Minimal workflow

This lesson is intentionally short on commands.

### Step 1: Read the pattern guidance

Start with the pattern pages:

- [Adapt-and-Apply](./architectures/adapt-and-apply.md)
- [Retrieval-Grounded Knowledge Assistant](./architectures/retrieval-grounded-assistant.md)
- [Evaluate-and-Improve Loop](./architectures/evaluate-and-improve-loop.md)
- [High-Throughput Inference Factory](./architectures/high-throughput-inference-factory.md)

### Step 2: Read the worked example

Use:

- [Customer Scenario](./worked-example/customer-scenario.md)
- [Recommended Architecture](./worked-example/recommended-architecture.md)

This example matters because it shows:

- the use case
- the constraints
- the candidate patterns considered
- why one was chosen
- what stayed out of scope

### Step 3: Produce your own architecture brief

Fill:

- [Architecture Brief Template](./templates/architecture-brief-template.md)

Your brief should include:

- selected pattern
- why the alternatives were rejected
- data flow
- compute placement
- evaluation gate
- pilot boundary

### Step 4: Validate the brief structure

Command:

```bash
python assets/validate_architecture_brief.py --brief /path/to/your/architecture-brief.md
```

Expected result:

- `VALIDATION_OK=1`
- all required sections are present

This is structural success.

It means the brief is complete enough to review.

It does not mean the architecture choice is good.

## How to read a good architecture brief

A good brief does three things clearly:

- identifies the dominant problem
- chooses one pattern that addresses it directly
- constrains the pilot so unnecessary components stay out

What you should look for:

- one explicit chosen pattern
- one explicit reason the other patterns were not chosen
- one explicit evaluation gate
- one explicit pilot boundary
- clear compute placement rather than “everything on LUMI-G”

## What a successful lesson outcome demonstrates

If this lesson works well, you have shown that:

- a use case can be mapped to a known architecture shape
- the architecture choice can be justified rather than guessed
- operational and evaluation concerns are built into the design
- the first pilot can remain constrained

That is different from saying the chosen architecture is final.

Reference architectures are starting points for disciplined pilots, not permanent system contracts.

## What to change next

After the first architecture brief, change one thing at a time.

Recommended order:

1. Narrow the pilot boundary before adding components.
2. Strengthen the evaluation gate before scaling complexity.
3. Reconsider retrieval vs adaptation if the chosen pattern feels overloaded.
4. Add sensitive-data redesign only if the data requires it.

## Cross-lesson map

Use the earlier lessons when the architecture choice points there:

- Lesson 01:
  choose this path when adaptation is the real need
- Lesson 03:
  choose this path when evolving knowledge and evidence-grounded answers dominate
- Lesson 04:
  use this when the architecture needs a controlled evaluation gate
- Lesson 05:
  use this when improvement requires a data-centric loop
- Lesson 07:
  use this when repeated inference operating pattern is part of the design
- Lesson 09:
  apply this when the chosen architecture must be redesigned for sensitive data or stronger trust constraints

## Next lesson

Next extension lesson: sensitive data and trustworthy operations.
