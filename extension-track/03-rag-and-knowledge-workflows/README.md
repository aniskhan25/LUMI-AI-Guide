# 03. Retrieval-Augmented Generation

## Goal

Build a small grounded RAG workflow on LUMI and inspect how evidence flows from documents to answers.

By the end of this lesson, you should be able to:

- explain what retrieval and generation each do in a RAG pipeline
- run a batch RAG workflow on LUMI
- validate that chunks, embeddings, retrieval results, and answers stay aligned
- make one safe retrieval-oriented modification

The practical question in this lesson is:

When should I use RAG instead of relying only on direct generation or model adaptation?

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You already know how to run Python and submit a batch job on LUMI.
- `../../env.sh` is configured with a valid `CONTAINER`.

## Working directory

Run commands in this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/03-rag-and-knowledge-workflows
```

## What RAG means here

This lesson uses three ideas together:

- chunking: split documents into smaller retrievable passages
- retrieval: find the most relevant chunks for a query
- generation: write an answer from the retrieved evidence

RAG combines retrieval and generation so answers stay grounded in a document corpus rather than relying only on model memory.

Use RAG when:

- answers must be grounded in a changing corpus
- evidence traceability matters
- adapting the model would be unnecessary or too expensive

Do not use this lesson to learn model serving or vector database operations. It is about the core grounded-answer pattern.

## RAG vs other patterns

- direct generation:
  useful when the model can answer from its own knowledge and traceable evidence is not required
- model adaptation:
  useful when you need the model itself to learn a task or style, not when you mainly need document grounding
- RAG:
  useful when the answer should come from external documents and those documents may change over time

## Why this baseline looks this way

The lesson uses:

- a small curated corpus of policy- and operations-style documents
- a matching query set with answers that should be recoverable from the corpus
- a simple local index rather than a vector database

This keeps the main question clear:

Can I preserve IDs across chunking, embeddings, retrieval, and answers, and can I trace each answer back to retrieved evidence?

## Main quality levers

The main choices that control RAG behavior in this lesson are:

- chunk size and overlap:
  too small and the evidence loses context; too large and retrieval becomes less precise
- retrieval `top_k`:
  too low and the right evidence may be missed; too high and the answer step may see too much noise
- embedding model:
  retrieval quality depends on how well the chunk and query vectors capture meaning
- evidence-to-answer handoff:
  even with good retrieval, the answer can still be weak if it does not stay close to the retrieved evidence

## Minimal workflow

The main path has three steps:

1. prepare the corpus
2. run the batch pipeline
3. validate the artifacts

Load the lesson runtime in your shell:

```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch
source ../../env.sh
```

### Step 1: Prepare the corpus

Command:

```bash
python data/prepare_corpus.py --output data
```

This writes:

- `data/corpus.jsonl`
- `data/queries.jsonl`

The corpus rows look like:

```json
{"doc_id":"doc-001","title":"Cooling System Maintenance","text":"...","metadata":{"domain":"operations","version":"v1"}}
```

### Step 2: Submit the RAG run

Command:

```bash
sbatch jobs/run_rag_single_node.sh
```

This batch job runs:

- chunking
- chunk embeddings
- local index build
- retrieval and answer generation

Outputs are written to:

```bash
outputs/rag-baseline
```

### Step 3: Validate outputs

Command:

```bash
python scripts/validate_rag_run.py --config configs/rag.yaml
```

Expected result:

- the Slurm log shows `GPU_VISIBLE_COUNT=1` or greater and `RUN_COMPLETE=1`
- `VALIDATION_OK=1`
- every query has retrieval results
- every answer has `evidence_chunk_ids` that point to real chunks

This is structural success. It means the RAG pipeline ran correctly and the artifacts are aligned.

It does not yet mean the answers are good.

Expected answer schema:

```json
{"query_id":"q-001","query":"...","answer":"...","evidence_chunk_ids":["doc-002-c0000"],"generation_backend":"hf_causal"}
```

## What this successful baseline demonstrates

If the lesson works end to end, you have shown that:

- documents can be chunked into stable retrievable units
- embeddings and chunk IDs stay aligned
- retrieval returns traceable evidence for each query
- generated answers can be tied back to specific chunk IDs

That is the lesson outcome. The commands are only the mechanism.

## How to diagnose a weak answer

When an answer looks weak, inspect `retrieval_results.jsonl` and `answers.jsonl` together.

Ask these questions in order:

1. Was the right chunk retrieved?
2. Was enough evidence retrieved?
3. Did the answer stay within the retrieved evidence?
4. Is the failure mainly retrieval, generation, or both?

Use this lesson rule:

If the retrieved evidence is wrong, fix retrieval before touching the answer model.

In practice:

- wrong chunk retrieved:
  revisit chunking, embedding choice, or `top_k`
- right chunk retrieved but weak answer:
  inspect the evidence-to-answer step before changing retrieval
- too many noisy chunks:
  reduce `top_k` or tighten chunking so the answer sees less irrelevant context

## What to change next

After the first successful run, change one thing at a time.

Recommended order:

1. Increase or decrease `retrieval.top_k`.
2. Change `chunking.chunk_words` and `chunking.overlap_words`.
3. Replace the corpus while preserving the same JSONL keys.
4. Swap the embedding model before changing the answer model.

## Troubleshooting

- `GPU_VISIBLE_COUNT=0`: check the partition, container, and runtime setup before debugging the model code.
- `VALIDATION_OK=1` is missing: inspect whether chunk IDs and embedding IDs match before looking at answer quality.
- weak answers: inspect `retrieval_results.jsonl` first. In RAG, bad retrieval usually matters more than prompt wording.

## Next lesson

Next extension lesson: evaluation and trustworthiness for grounded AI workflows.
