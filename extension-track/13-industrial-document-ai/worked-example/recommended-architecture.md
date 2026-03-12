# Worked Example: Recommended Architecture

## Selected Pattern

Retrieval-grounded technical assistant with explicit evidence output.

Core flow:

1. ingest approved manuals/procedures with revision metadata
2. build embeddings and retrieval index
3. retrieve top-k evidence passages for each query
4. generate grounded answer constrained by evidence
5. attach review flag when evidence is weak or missing

## Placement

- compute-heavy embedding and generation on LUMI-G
- shared corpus and promoted artifacts staged via LUMI-O when needed
- containerized runtime using AI Software Environment lineage
