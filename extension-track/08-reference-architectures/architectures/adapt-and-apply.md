# Architecture A: Adapt-and-Apply

## Best For

- Domain-specific classification, extraction, ranking, or generation where base models need targeted adaptation.

## Core Flow

dataset -> adaptation on LUMI-G -> evaluation gate -> batch/service inference

## Required Components

- Baseline dataset and versioned splits
- Adaptation training workflow
- Evaluation benchmark and quality gate
- Inference packaging path (batch or service-style)

## Compute Placement

- LUMI-G: adaptation and high-throughput inference
- lighter orchestration/data preparation: CPU-side or external helper environment

## Data Pattern

- versioned train/eval artifacts
- optional staged dataset handoff through LUMI-O
- optional curated data source via Dataset-as-a-Service

## Evaluation Gate

Minimum gate before wider pilot:

- accuracy/quality target met on benchmark slice
- failure categories reviewed
- inference output schema validated

## First Risk To Watch

- Over-adapting when retrieval-grounded pattern would solve the knowledge problem faster.

## Operational Checklist

- adaptation objective explicitly defined
- dataset lineage and versioning recorded
- benchmark gate attached to model release
- inference path chosen and validated

