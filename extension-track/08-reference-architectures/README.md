# 08. Reference Architectures for Customer AI Systems on LUMI AI Factory

This capstone lesson teaches how to select and document end-to-end customer AI system patterns using the workflows built in Lessons 1-7.

## What This Lesson Enables

Choose and justify a customer-ready AI system pattern by producing:

- architecture selection rationale
- data flow and compute placement
- operational checkpoints
- evaluation gate
- constrained pilot scope

## The Reference-Architecture Mindset

Use a small set of proven patterns instead of designing from scratch every time:

use case -> constraints -> pattern choice -> operational plan -> evaluation gate

## When To Use Each Architecture

Use the decision matrix in [templates/decision-matrix.md](templates/decision-matrix.md) to select among four patterns:

- Adapt-and-Apply
- Retrieval-Grounded Knowledge Assistant
- Evaluate-and-Improve Loop
- High-Throughput Inference Factory

## The Four Architectures At A Glance

Catalog pages:

- [Adapt-and-Apply](architectures/adapt-and-apply.md)
- [Retrieval-Grounded Knowledge Assistant](architectures/retrieval-grounded-assistant.md)
- [Evaluate-and-Improve Loop](architectures/evaluate-and-improve-loop.md)
- [High-Throughput Inference Factory](architectures/high-throughput-inference-factory.md)

## Worked Example Architecture Brief

Read:

- [Customer Scenario](worked-example/customer-scenario.md)
- [Recommended Architecture](worked-example/recommended-architecture.md)

This worked example shows a complete architecture decision from use case to pilot boundary.

## Why This Fits LUMI AI Factory

This lesson keeps platform placement explicit:

- compute-heavy model stages on LUMI-G
- AI runtime in AI Factory container environment
- staged/shared corpus and outputs through LUMI-O when needed
- curated datasets through Dataset-as-a-Service when relevant

## Operational Checkpoints

Use [assets/operational-checklists.md](assets/operational-checklists.md) to gate early pilot execution for each pattern.

## Common Mis-Architectures

See [troubleshooting/common-misarchitectures.md](troubleshooting/common-misarchitectures.md).

## Produce Your Own Architecture Brief

Fill [templates/architecture-brief-template.md](templates/architecture-brief-template.md) for your customer use case.  
Optionally validate completeness with:

```bash
python assets/validate_architecture_brief.py --brief /path/to/your/architecture-brief.md
```

## Operational Checklist

- Use case defined
- Pattern chosen with explicit rationale
- Compute-heavy stages mapped to LUMI-G
- Data storage/sharing path selected
- Evaluation gate defined
- First pilot scope constrained

## Closing The Track

This lesson is the synthesis point: it turns all earlier workflow capabilities into repeatable, customer-ready solution blueprints.

