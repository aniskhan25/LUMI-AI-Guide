# 03. Retrieval-Augmented Generation

## Goal

Build a small grounded RAG workflow on LUMI and inspect how evidence flows from documents to answers.

By the end of this lesson, you should be able to:

- explain what retrieval and generation each do in a RAG pipeline
- run a batch RAG workflow on LUMI
- validate that chunks, embeddings, retrieval results, and answers stay aligned
- make one safe retrieval-oriented modification

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

## Why this baseline looks this way

The lesson uses:

- a small curated corpus of policy- and operations-style documents
- a matching query set with answers that should be recoverable from the corpus
- a simple local index rather than a vector database

This keeps the main question clear:

Can I preserve IDs across chunking, embeddings, retrieval, and answers, and can I trace each answer back to retrieved evidence?

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
