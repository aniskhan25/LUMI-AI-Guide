# Architecture C: Evaluate-and-Improve Loop

## Best For

- Customer pilots where quality, trustworthiness, and iterative improvement are the primary concerns.

## Core Flow

baseline workflow -> benchmark -> failure analysis -> targeted improvement (synthetic/workflow changes) -> rerun -> compare

## Required Components

- Versioned evaluation set
- Deterministic scoring and failure extraction
- Improvement mechanism (data-centric or workflow-centric)
- Before/after comparison report

## Compute Placement

- LUMI-G: repeated generation, evaluation, and improvement iterations
- lighter report and artifact aggregation can run off-GPU

## Data Pattern

- strict benchmark versioning
- synthetic additions with explicit provenance
- optional artifact staging via LUMI-O

## Evaluation Gate

Minimum gate before wider pilot:

- controlled variant comparison completed
- no regression on critical slices
- failure categories reviewed and documented

## First Risk To Watch

- Expanding architecture complexity before establishing baseline quality evidence.

## Operational Checklist

- benchmark set fixed and versioned
- IDs stable across all artifacts
- failure review mandatory in loop
- comparison decision criteria documented

