# Development Spec Template

Use this document to lock decisions and track implementation for Lesson 03.

## 1. Lesson Identity

- Lesson id: `EXT-03`
- Title: `Retrieval-Augmented Generation on LUMI-G with AI Factory Data and Software Services`
- Short nav title: `RAG on MI250X`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

Fill this table before editing lesson prose or scripts.

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Sample corpus | manuals, policy docs, abstracts, technical notes | `TBD` | `TBD` | `TBD` | `TBD` |
| Chunking strategy | fixed words, fixed tokens, semantic chunks | `TBD` | `TBD` | `TBD` | `TBD` |
| Embedding model | e.g. E5/BGE/DistilBERT-family | `TBD` | `TBD` | `TBD` | `TBD` |
| Retriever implementation | cosine matrix, FAISS, managed DB | `TBD` | `TBD` | `TBD` | `TBD` |
| Answer schema | answer + evidence IDs + excerpts | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- Corpus preparation and chunking
- Embedding generation
- Retriever index build
- Top-k retrieval and grounded generation
- Output validation with evidence-preserving schema

### Out Of Scope

- Online serving and APIs
- Full production vector database operations
- Agent frameworks and orchestration systems
- Deep ranking/evaluation methodology
- Generic cluster tutorials

## 4. Learning Outcomes

By lesson end, learner can:

1. build a runnable baseline RAG workflow on LUMI-G
2. explain why chunking and identifiers are load-bearing design choices
3. run retrieval and grounded generation with traceable evidence
4. validate artifact consistency across corpus/chunks/embeddings/answers
5. perform one safe modification (chunk size or top-k) and rerun

## 5. Lesson Structure Contract

The published lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal working example
- F. Verification
- G. Why this works on LUMI-G
- H. Data and storage considerations
- I. Common failure modes
- J. Extension paths
- K. Operational checklist
- L. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Corpus preparation script | Yes | `scripts/prepare_corpus.py` |
| Chunking script | Yes | `scripts/chunk_corpus.py` |
| Embedding script | Yes | `scripts/embed_chunks.py` |
| Index build script | Yes | `scripts/build_index.py` |
| Query + generation script | Yes | `scripts/answer_queries.py` |
| Validation script | Yes | `scripts/validate_rag_run.py` |
| Canonical batch jobscript | Yes | `jobs/run_rag_single_node.sh` |
| Sample corpus and query set | Yes | `data/sample_corpus.jsonl`, `data/sample_queries.jsonl` |
| Troubleshooting page | Yes | `troubleshooting/common-failures.md` |

## 7. Content Acceptance Criteria

- Lesson reads top-to-bottom without outside explanation
- Golden path has minimal branching
- Success conditions and artifact paths are explicit
- At least three common failure modes are documented
- Retrieval and generation roles are operationally distinct

## 8. Technical Acceptance Criteria

- User can run with minor path/account edits
- Pipeline writes chunk, embedding, retrieval, and answer artifacts
- Validation checks cross-artifact consistency
- GPU use is confirmable for embedding/generation steps
- Example is stable for repeated teaching use

## 9. Pedagogical Acceptance Criteria

- Teaches one operational capability: baseline grounded RAG workflow
- Learner can explain why retrieval quality affects answer quality
- Learner can perform one controlled parameter modification
- Learner leaves with a reusable corpus-grounded pattern

## 10. Testing Plan

Record outputs for each gate:

| Gate | Command | Pass/Fail | Notes |
|---|---|---|---|
| Prepare corpus | `python scripts/prepare_corpus.py --output data` | `TBD` | `TBD` |
| Chunk corpus | `python scripts/chunk_corpus.py --config configs/rag.yaml` | `TBD` | `TBD` |
| Embed chunks | `python scripts/embed_chunks.py --config configs/rag.yaml` | `TBD` | `TBD` |
| Build index | `python scripts/build_index.py --config configs/rag.yaml` | `TBD` | `TBD` |
| Answer queries | `python scripts/answer_queries.py --config configs/rag.yaml` | `TBD` | `TBD` |
| Validate run | `python scripts/validate_rag_run.py --config configs/rag.yaml` | `TBD` | `TBD` |
| LUMI run | `sbatch jobs/run_rag_single_node.sh` | `TBD` | `TBD` |

## 11. Review Sign-Off

| Role | Name | Status | Date | Notes |
|---|---|---|---|---|
| Content reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Technical reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Platform reviewer (LUMI-G + data services) | `TBD` | `Pending` | `TBD` | `TBD` |
