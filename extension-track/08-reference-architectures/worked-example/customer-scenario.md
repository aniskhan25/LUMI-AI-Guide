# Worked Example: Customer Scenario

## Scenario

An engineering organization wants an internal assistant for technical manuals and incident runbooks.  
The team needs grounded answers with evidence references and cannot accept unsupported responses for high-priority operations queries.

## Constraints

- Corpus updates weekly.
- Answers must include traceable evidence references.
- Pilot must launch within six weeks.
- Initial usage is internal team support, not public endpoint traffic.

## Current State

- Baseline prompt-only generation gives inconsistent factual answers.
- No retrieval layer exists yet.
- No structured evaluation gate is in place.

## Decision Goal

Select the minimum viable architecture that can produce grounded answers, be evaluated quickly, and scale to higher request volumes later.

