# Worked Example: Sharing Model

## Internal vs Cross-Project Sharing

Project-internal:

- active experiments and working copies remain in the project workspace
- all project members can access project-owned data

Cross-project collaboration:

- publish read-only dataset prefix for collaboration project
- publish promoted artifact bundle prefix for consuming team
- avoid exposing full bucket when only selected prefixes are needed

## DaaS Positioning

When a curated upstream dataset exists through Dataset-as-a-Service:

- consume DaaS dataset as source-of-truth input
- derive project-specific working copies for experiments
- do not replicate DaaS into uncontrolled parallel copies

## Handoff Package

Each cross-project handoff includes:

- promoted artifact ID
- source run ID
- evaluation summary
- known limitations
- retrieval path and access scope
