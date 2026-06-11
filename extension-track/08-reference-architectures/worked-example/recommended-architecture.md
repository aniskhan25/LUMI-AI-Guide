# Worked Example: Recommended Architecture Brief

## Selected Pattern

Architecture B: Retrieval-Grounded Knowledge Assistant

## Why This Pattern Fits

- The core risk is factual drift from prompt-only generation.
- The corpus changes regularly, making retrieval preferable to immediate adaptation.
- Stakeholders require evidence-aware answers.

## Why Alternatives Were Not Chosen

- Adapt-and-Apply: adaptation alone does not solve frequent corpus updates.
- Evaluate-and-Improve Loop: needed as a gate, but not the primary architecture shape.
- High-Throughput Inference Factory: useful later for volume, but does not provide grounding by itself.

## Data Flow

corpus ingestion -> chunking -> embedding/index build -> retrieval -> grounded generation -> evaluated outputs

## Compute Placement

- LUMI-G: chunk embedding, retrieval-assisted generation, high-throughput evaluation runs
- Lighter orchestration: index metadata management and report assembly

## Data/Storage Integration

- Corpus snapshots stored and versioned in project storage
- Optional staging/sharing through LUMI-O for cross-team collaboration
- Optional curated input flows from Dataset-as-a-Service

## Evaluation Gate

- Retrieval hit-rate threshold on benchmark set
- Groundedness threshold (answer supported by retrieved evidence)
- Failure categories reviewed for high-priority operations queries

## First Pilot Scope

- One domain subset of manuals and runbooks
- Internal-only users
- Batch-style or service-style inference in scheduled LUMI-G windows
- No public endpoint exposure in pilot phase

## Operational Checkpoints

- Stable chunk and evidence IDs
- Structured request/response logging with IDs
- Weekly benchmark rerun against updated corpus
- Comparison report before each pilot expansion

