# 13. Industrial Document AI

## Goal

Design a trustworthy technical document assistant where answers must be evidence-linked, revision-aware, and safe for operational use.

By the end of this lesson, you should be able to:

- explain how technical-document workflows differ from generic RAG
- define the metadata and corpus rules needed for revision-aware retrieval
- define an answer contract that is evidence-linked and fail-safe
- attach evaluation and update ownership rules to the workflow
- produce a reusable domain blueprint for a technical knowledge workflow

The practical question in this lesson is:

How should I design a trustworthy technical document assistant when answers must be evidence-linked, revision-aware, and safe for operational use?

## Assumptions

- You completed [03. Retrieval-Augmented Generation](../03-rag-and-knowledge-workflows/README.md).
- You completed [04. Evaluation and Trustworthiness](../04-evaluation-and-trustworthiness/README.md).
- You completed [08. Reference Architectures](../08-reference-architectures/README.md).
- Preferred: you also completed [10. MLOps and Lifecycle Management](../10-mlops-and-lifecycle-management/README.md) and [11. Team Operating Models and Collaboration](../11-team-operating-models-and-collaboration/README.md).

## Working directory

Use this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/13-industrial-document-ai
```

## What the domain-specific concepts mean here

- authoritative source:
  the document system that defines valid operational guidance
- revision-aware retrieval:
  answers must point to the exact revision used, not just a document family
- evidence-linked answer:
  the answer is only acceptable if it includes supporting document and chunk IDs
- unsupported-answer handling:
  the workflow must fail safely when evidence is weak or missing

This lesson is not introducing a new core AI pattern. It is showing how earlier patterns must change in a technical-document domain.

## When this lesson is needed

Use this lesson when:

- maintenance or operations workflows depend on manuals and procedures
- draft and approved guidance must not be mixed
- answers need evidence and revision linkage
- failure can create operational, quality, or safety risk

Typical tasks include:

- troubleshooting specific alarms
- answering maintenance-interval questions
- checking prerequisites, constraints, or safety notes
- finding the correct procedure step for an operational condition

## What this lesson is and is not

This lesson is:

- a domain accelerator for technical knowledge workflows
- a way to adapt grounded retrieval to revision-sensitive documents
- a way to add domain-specific answer and review rules

This lesson is not:

- a general-purpose chat assistant pattern
- a search-only interface design
- a full industrial knowledge platform
- a sector-specific compliance framework

## How this lesson fits the track

This lesson composes earlier work:

- Lesson 03:
  grounded retrieval and evidence flow
- Lesson 04:
  evaluation and failure analysis
- Lesson 08:
  architecture selection
- Lessons 10 and 11:
  lifecycle and team operating rules

Lesson 13 applies those patterns to one domain where metadata quality, revision control, and answer discipline matter more than fluency.

## Main design levers

The choices that matter most in this lesson are:

- document types included:
  manuals, procedures, reports, bulletins
- approved versus draft boundary:
  what content is allowed to influence operational answers
- revision and approval metadata:
  what must be tracked at document and chunk level
- sequence preservation:
  whether procedure order must survive chunking and answer generation
- answer schema:
  what evidence and review fields every answer must include
- update cadence:
  when re-indexing and re-evaluation are required

These levers matter more than simply “using RAG.”

## How to design this workflow safely

Use this order:

1. Pick one authoritative document source.
2. Make revision and approval state explicit.
3. Preserve sequence metadata for procedures.
4. Keep answer support and answer fluency separate.
5. Define unsupported-answer behavior before rollout.
6. Attach update ownership and promotion rules.

Use this lesson rule:

No delivery-grade answer without evidence IDs, revision linkage, and defined unsupported-answer behavior.

## How to spot a weak domain design

Warning signs include:

- answers cite document ID but not revision
- draft and approved documents are mixed in retrieval
- procedure steps lose order during chunking or generation
- fluency is accepted without technical support
- update ownership is unclear when source documents change

A weak design often sounds convincing while failing at traceability and operational safety.

## Minimal workflow

This lesson is short on commands because the work is in the design.

### Step 1: Study the worked example

Read:

- [Scenario](./worked-example/scenario.md)
- [Recommended architecture](./worked-example/recommended-architecture.md)
- [Corpus design](./worked-example/corpus-design.md)
- [Evaluation and review](./worked-example/evaluation-and-review.md)

These show:

- the operational use case
- the chosen architecture pattern
- the corpus metadata rules
- the review and promotion gate

### Step 2: Fill the domain blueprint

Use:

- [Domain use-case brief](./templates/domain-use-case-brief.md)
- [Technical corpus schema](./templates/technical-corpus-schema.md)
- [Evaluation checklist](./templates/evaluation-checklist.md)
- [Update operating model](./templates/update-operating-model.md)

The use-case brief should make explicit:

- the users
- the document scope
- the output constraints
- the architecture choice

The corpus schema should make explicit:

- document and chunk identifiers
- revision linkage
- approval state
- sequence metadata for procedures

The evaluation checklist should make explicit:

- support and correctness checks
- unsupported-answer handling
- failure taxonomy

The update model should make explicit:

- ownership
- update triggers
- promotion rules

### Step 3: Validate the blueprint structure

Command:

```bash
python assets/validate_domain_blueprint.py \
  --use-case templates/domain-use-case-brief.md \
  --schema templates/technical-corpus-schema.md
```

Expected result:

- `VALIDATION_OK=1`
- the required use-case and schema sections are present

This is structural success.

It means the blueprint is complete enough to review.

It does not mean the corpus model or answer behavior is safe enough.

## How to read a good technical-document blueprint

A good blueprint should answer these questions quickly:

- what is the authoritative source?
- how are approved and draft documents separated?
- how is revision tracked?
- what evidence fields must every answer include?
- what happens when evidence is weak?
- who owns updates and promotion?

If those answers are not obvious, the workflow is still too fragile for technical use.

## Revision and update note

For the specific problem of handling revised documents, keep:

- [Revision handling note](./assets/revision-handling-note.md)

That note is worth keeping separate because revision handling is one of the main ways technical assistants drift into unsafe behavior.

## What this lesson outcome demonstrates

If this lesson works well, you have shown that:

- the workflow is grounded in an authoritative technical corpus
- revision and approval state are explicit
- answers can be evidence-linked and fail safe
- technical evaluation is stronger than fluency-only review
- update ownership is part of the design

That is different from proving the domain assistant is ready for all real-world deployment contexts.

## What to change next

After the first domain blueprint, change one thing at a time.

Recommended order:

1. tighten approved-versus-draft boundaries before widening the corpus
2. strengthen answer schema and unsupported-answer handling before broadening users
3. improve evaluation slices before increasing automation
4. strengthen update ownership before increasing document refresh frequency

## Next lesson

This is the current endpoint of the extension track.
