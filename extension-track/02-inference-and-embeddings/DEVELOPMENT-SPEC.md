# Development Spec Template

Use this document to lock decisions and track implementation for Lesson 02.

## 1. Lesson Identity

- Lesson id: `EXT-02`
- Title: `Efficient Inference and Embedding Pipelines on LUMI-G`
- Short nav title: `Inference and embeddings on MI250X`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

Fill this table before changing lesson prose or scripts.

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Primary task | Embeddings, classification, generation | `Embeddings` | Stable and easy to validate | `TBD` | `TBD` |
| Model family | e.g. BGE-small, E5-small, DistilBERT, Qwen embed model | `TBD` | `TBD` | `TBD` | `TBD` |
| Output format | JSONL vectors, JSONL + binary array sidecar, parquet | `TBD` | `TBD` | `TBD` | `TBD` |
| Validation contract | Count + IDs + vector dim + non-empty checks | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- One end-to-end batch embedding workflow on LUMI-G
- One optional generation variant using the same pattern
- One recommended AI Factory container-first execution path
- Output schema and explicit validation contract

### Out Of Scope

- Online serving and endpoint operations
- Full RAG system design
- Vector database setup
- Distributed inference orchestration at scale
- Generic Slurm or benchmarking tutorials

## 4. Learning Outcomes

By lesson end, learner can:

1. run batch inference and embedding generation on LUMI-G
2. explain training-oriented vs inference-oriented execution
3. choose a sensible batch size strategy
4. validate output completeness and schema consistency
5. perform one safe throughput-oriented modification

## 5. Lesson Structure Contract

The published lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal working example
- F. Verification
- G. Throughput thinking on LUMI-G
- H. Common failure modes
- I. Extension paths
- J. Operational checklist
- K. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Primary embeddings script | Yes | `scripts/run_embeddings.py` |
| Optional generation script | Yes (optional execution) | `scripts/run_generation.py` |
| Config files | Yes | `configs/embeddings.yaml`, `configs/generation.yaml` |
| Single-node jobscript | Yes | `jobs/run_embeddings_single_node.sh` |
| Validation script | Yes | `scripts/validate_outputs.py` |
| Sample input datasets | Yes | `data/sample_corpus.jsonl`, `data/sample_generation_inputs.jsonl` |
| Output schema doc | Yes | `assets/expected-output-schema.md` |
| Troubleshooting page | Yes | `troubleshooting/common-failures.md` |

## 7. Content Acceptance Criteria

- Lesson runs top-to-bottom without outside explanation
- Main path has minimal branching
- Success artifacts and expected file paths are explicit
- At least three common failure modes documented
- Practical distinction between embeddings and generation included

## 8. Technical Acceptance Criteria

- Main example runs with minor path/account edits
- Structured output is written for all inputs
- Validation checks completeness and schema consistency
- GPU use can be confirmed from logs/summary
- Pipeline is stable for repeated teaching use

## 9. Pedagogical Acceptance Criteria

- Teaches one operational capability: model-as-pipeline execution
- Learner can explain why batching matters
- Learner can modify one safe execution parameter
- Learner can reuse the pattern for new corpora

## 10. Testing Plan

Record outputs for each gate:

| Gate | Command | Pass/Fail | Notes |
|---|---|---|---|
| Data prep | `python data/prepare_sample_data.py --output data` | `TBD` | `TBD` |
| Embeddings local run | `python scripts/run_embeddings.py --config configs/embeddings.yaml` | `TBD` | `TBD` |
| Embeddings validation | `python scripts/validate_outputs.py --mode embeddings ...` | `TBD` | `TBD` |
| Generation local run (optional) | `python scripts/run_generation.py --config configs/generation.yaml` | `TBD` | `TBD` |
| Generation validation (optional) | `python scripts/validate_outputs.py --mode generation ...` | `TBD` | `TBD` |
| LUMI embeddings run | `sbatch jobs/run_embeddings_single_node.sh` | `TBD` | `TBD` |

## 11. Review Sign-Off

| Role | Name | Status | Date | Notes |
|---|---|---|---|---|
| Content reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Technical reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Platform reviewer (LUMI-G inference focus) | `TBD` | `Pending` | `TBD` | `TBD` |

