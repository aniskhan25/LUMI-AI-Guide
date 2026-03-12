# Architecture B: Retrieval-Grounded Knowledge Assistant

## Best For

- Document QA, grounded summarization, and evidence-aware generation over evolving corpora.

## Core Flow

corpus -> chunking -> embeddings -> retrieval -> grounded generation -> evaluation gate

## Required Components

- Corpus ingestion and chunk manifest
- Embedding pipeline and retriever index
- Prompt assembly with retrieved evidence
- Output schema with evidence IDs
- Evaluation slice for retrieval and groundedness

## Compute Placement

- LUMI-G: embeddings and generation-heavy steps
- lightweight retrieval/index orchestration can remain CPU-side in pilot scope

## Data Pattern

- Corpus and chunk IDs must remain stable
- LUMI-O can stage/share corpora and outputs
- Dataset-as-a-Service can provide managed curated corpora

## Evaluation Gate

Minimum gate before wider pilot:

- retrieval hit/recall threshold met
- groundedness threshold met
- evidence traceability present in outputs

## First Risk To Watch

- Treating generation quality as model-only quality while retrieval quality is weak.

## Operational Checklist

- chunking strategy documented
- retrieval metrics tracked separately from answer metrics
- evidence IDs preserved in final outputs
- failure taxonomy includes retrieval vs generation errors

