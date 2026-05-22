# 09. Sensitive Data and Trustworthy Operations

## Goal

Redesign a baseline AI workflow so it remains usable when the data or outputs become sensitive enough that the original architecture is no longer acceptable.

By the end of this lesson, you should be able to:

- explain when a baseline workflow must be redesigned for sensitive data
- identify the stages where exposure risk is highest
- apply minimization, pseudonymization, and logging discipline to the architecture
- define a trust gate before wider pilot use
- produce a revised architecture brief that makes the new boundaries explicit

The practical question in this lesson is:

How should I redesign an AI workflow when the data or outputs become sensitive enough that the baseline architecture is no longer acceptable?

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You completed [08. Reference Architectures](../08-reference-architectures/README.md).
- You already have one baseline architecture to modify.

## Working directory

Use this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/09-sensitive-data-and-trustworthy-operations
```

## What the core ideas mean here

- sensitive data:
  content or identifiers that change what handling is acceptable
- minimization:
  remove non-essential fields before model-facing stages
- pseudonymization:
  separate identifiers from content where possible and replace direct IDs with stable surrogates
- trust gate:
  an explicit review checkpoint before outputs are used more broadly

This lesson is about operational redesign, not model retraining.

## When this lesson is needed

Use this lesson when the workflow touches:

- personal data or identifying attributes
- customer-confidential documents
- restricted internal corpora
- outputs whose failure has higher operational or trust consequences

Typical triggers are:

- the baseline architecture is technically functional but exposes too much information
- prompts, retrieved context, or logs now contain material that requires tighter handling
- pilot expansion is blocked until review and control points are explicit

## What this lesson is and is not

This lesson is:

- a redesign tutorial
- a way to reduce exposure in an existing workflow
- a way to make trust review operational instead of implied

This lesson is not:

- a legal or compliance determination
- a full security architecture
- a substitute for organization-specific policy
- proof that the workflow is safe in every deployment context

## How this lesson fits the track

Lesson 08 chose the baseline architecture.

Lesson 09 changes the question from:

- which architecture fits this use case?

to:

- how must that architecture change once the data or outputs become sensitive?

Lesson 10 then builds on this by handling lifecycle, promotion, and operational management over time.

## When pseudonymization is not enough

Pseudonymization helps, but it does not automatically make a workflow non-sensitive.

It is not enough when:

- the corpus itself is sensitive even after identifiers are replaced
- re-identification remains plausible through context or joining
- prompts or retrieved passages still carry restricted content
- outputs can still leak sensitive information
- logs retain raw text that should not be there

Use this lesson rule:

Pseudonymization reduces one class of exposure. It does not remove the need for minimization, logging discipline, and review gates.

## Main redesign levers

The changes that matter most in this lesson are:

- data classification by stage:
  what is red, amber, or green at each step
- content versus identifier separation:
  what must stay together and what must be split
- prompt and retrieval minimization:
  what the model actually needs to see
- log minimization:
  IDs, status, and timing by default; raw text only when explicitly justified
- artifact retention scope:
  fewer sensitive copies and shorter retention
- trust gate:
  explicit owner, checks, and pass/fail decision before expansion
- tighter pilot boundary:
  the first pilot should usually become narrower, not broader

## How to redesign safely

Use this order:

1. Classify the workflow by stage.
2. Identify the highest-exposure transitions.
3. Remove non-essential fields before model-facing stages.
4. Separate lookup tables from routine inference artifacts.
5. Reduce prompt, retrieval, and log content to what is actually needed.
6. Add a trust gate before broader downstream use.
7. Tighten the pilot boundary so risky expansion does not happen by default.

In practice, this means:

- classify first
- minimize before generation or retrieval
- separate reversible mappings from routine artifacts
- keep sensitive text out of routine logs
- review outputs before wider sharing

## How to spot an insufficient redesign

Warning signs include:

- the sensitive-data variant still produces nearly the same logs and artifacts as the baseline
- identifiers are pseudonymized but raw sensitive text still appears everywhere
- a trust gate exists on paper but has no owner, sample size, or pass criteria
- the architecture brief changed wording but not actual exposure boundaries
- the pilot scope stayed the same even though sensitivity increased

An insufficient redesign often looks organized on paper while leaving the real exposure pattern unchanged.

## Minimal workflow

This lesson is short on commands because the main work is architectural.

### Step 1: Study the baseline and the redesign

Read:

- [Baseline architecture](./worked-example/baseline-architecture.md)
- [Sensitive-data variant](./worked-example/sensitive-data-variant.md)
- [Data-flow map](./worked-example/data-flow-map.md)

These show:

- what the baseline architecture was
- why it became insufficient
- which stages were redesigned
- where controls were added

### Step 2: Fill the classification and trust artifacts

Use:

- [Data classification table](./templates/data-classification-table.md)
- [Trust gate template](./templates/trust-gate-template.md)
- [Revised architecture brief](./templates/revised-architecture-brief.md)

The classification table answers:

- what data appears at each stage
- how sensitive it is
- what is truly required
- what minimization or pseudonymization action is needed

The trust gate answers:

- which outputs are in scope
- which checks must pass
- who owns the decision

The revised brief captures the actual redesign.

### Step 3: Validate the revised brief structure

Command:

```bash
python assets/validate_revised_brief.py --brief /path/to/your/revised-architecture-brief.md
```

Expected result:

- `VALIDATION_OK=1`
- all required sections are present

This is structural success.

It means the brief is complete enough to review.

It does not mean the redesign is safe enough.

## How to read a successful redesign

A successful redesign should show:

- fewer sensitive artifacts than the baseline
- explicit classification by workflow stage
- clear separation of identifiers and content where possible
- reduced raw-text logging
- one trust gate before wider downstream use
- a narrower and more controlled pilot boundary

The redesign is successful only if the exposure pattern changed, not just the documentation.

## How to review the worked example

In the worked example, pay attention to:

- why the baseline Retrieval-Grounded Knowledge Assistant was still the right overall architecture
- why the redesign happened at the data-handling and review layers rather than by swapping architectures entirely
- which transitions were highest risk:
  ingestion to preprocessing, and retrieval to prompt build
- how the trust gate changes the rollout condition

That is the main lesson:

When sensitivity rises, you often keep the same architectural pattern but tighten the boundaries around data, artifacts, and review.

## Operational note on prompts and logs

For practical prompt and log minimization guidance, use:

- [Prompt and logging minimization note](./assets/prompt-and-logging-minimization.md)

That note is worth keeping separate because it is operationally specific rather than just explanatory.

## What this lesson outcome demonstrates

If this lesson works well, you have shown that:

- a baseline architecture can be redesigned rather than discarded
- sensitive stages can be identified explicitly
- exposure can be reduced through minimization, separation, and logging discipline
- a trust gate can be made operational
- the pilot boundary can be tightened to reflect the new risk

That is different from proving the workflow is fully safe or fully compliant.

## What to change next

After the first redesign, change one thing at a time.

Recommended order:

1. tighten the pilot boundary before adding new features
2. reduce logging and artifact retention before expanding sharing
3. strengthen the trust gate before increasing user scope
4. revisit the baseline architecture only if the redesign still leaves unacceptable exposure

## Next lesson

Next extension lesson: MLOps and lifecycle management for AI Factory workflows.
