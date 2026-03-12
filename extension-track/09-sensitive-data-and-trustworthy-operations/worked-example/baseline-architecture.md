# Worked Example: Baseline Architecture (Before Sensitive Redesign)

## Selected Baseline Pattern

Lesson 8 Architecture B: Retrieval-Grounded Knowledge Assistant

## Baseline Flow

internal corpus -> chunking -> embedding/index -> retrieval -> grounded generation -> outputs/logs

## Baseline Strengths

- grounded responses over changing corpus
- traceable evidence IDs
- clear retrieval/generation separation

## Baseline Weakness Under Sensitive Constraints

- prompts may include direct identifying fields
- retrieved contexts can retain unnecessary sensitive details
- logs may store raw request/response text longer than needed
- intermediate artifacts are not tightly scoped by sensitivity class

## Redesign Trigger

The use case now includes sensitive customer records embedded in internal documents and requires stricter handling and review before outputs are shared.

