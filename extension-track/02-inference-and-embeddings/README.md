# 02. Inference and Embeddings

## Goal

Run a model as a batch pipeline on LUMI-G and produce reusable embedding outputs.

By the end of this lesson, you should be able to:

- explain how inference on LUMI differs from training
- run a batch embedding pipeline on LUMI-G
- validate output integrity for a full corpus pass
- make one safe throughput-oriented modification

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You already know how to run Python and submit a batch job on LUMI.
- `../../env.sh` is configured with a valid `CONTAINER`.

## Working directory

Run commands in this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/02-inference-and-embeddings
```

## What is new in this lesson

This lesson treats the model as a data-processing pipeline rather than a training job.

The main path uses:

- a sentence embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- a small AG News subset as open input data
- JSONL input and JSONL output

The main question is no longer “did training run?” It is:

Can I process a corpus on LUMI-G, preserve record IDs, and write structured outputs that downstream systems can trust?

## Why embeddings are the main path

This lesson uses two common kinds of inference:

- embeddings: turn text into vectors so you can compare, search, cluster, or retrieve related text
- generation: produce new text from a prompt, such as a summary or completion

Embeddings are the main path because they are the cleaner first inference workload:

- the output format is easy to validate
- batching matters immediately
- the result is reusable for retrieval, clustering, and indexing

Generation is still useful, but it is supplemental in this lesson because it adds more model- and prompt-specific variability.

## Minimal workflow

The main path has three steps:

1. prepare input
2. run embeddings
3. validate outputs

Load the lesson runtime in your shell:

```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch
source ../../env.sh
```

### Step 1: Prepare the AG News subset

Command:

```bash
python data/prepare_ag_news.py --output data
```

This downloads a small AG News subset and writes:

- `data/ag_news_corpus.jsonl`
- `data/ag_news_generation_inputs.jsonl`

The corpus rows look like:

```json
{"id":"doc-0001","text":"Wall St. Bears Claw Back Into the Black","metadata":{"label":2,"category":"business"}}
```

### Step 2: Submit the embeddings run

Command:

```bash
sbatch jobs/run_embeddings_single_node.sh
```

This runs the embeddings pipeline inside one batch job.

Success signal in the Slurm output:

- `GPU_VISIBLE_COUNT=1` or greater
- `BATCH=... OUTPUT_RECORDS=...`
- `RUN_COMPLETE=1`

Outputs are written to:

```bash
outputs/embeddings-baseline
```

### Step 3: Validate outputs

Command:

```bash
python scripts/validate_outputs.py \
  --mode embeddings \
  --input-jsonl data/ag_news_corpus.jsonl \
  --output-jsonl outputs/embeddings-baseline/embeddings.jsonl \
  --summary-json outputs/embeddings-baseline/run_summary.json
```

Expected result:

- `VALIDATION_OK=1`
- matching input and output counts
- a consistent embedding dimension

## How to interpret the result

Look for these signals:

- `GPU_VISIBLE_COUNT` confirms the runtime saw GPUs.
- output count matching input count confirms a full corpus pass.
- preserved `id` values confirm safe downstream mapping.
- a consistent embedding dimension confirms one coherent embedding model wrote the whole file.
- `run_summary.json` confirms the run is inspectable and reusable.

Expected embeddings output schema:

```json
{"id":"doc-0001","embedding":[0.0123,-0.9987,0.1244],"metadata":{"label":2,"category":"business"}}
```

## What this successful baseline demonstrates

If the lesson works end to end, you have shown that:

- a model can process a corpus on LUMI-G without training
- batching works correctly for the chosen input size
- outputs preserve IDs and remain machine-readable
- the embedding pipeline is ready for downstream retrieval or analytics

## What to change next

After the first successful run, change one thing at a time.

Recommended order:

1. Increase `inference.batch_size` conservatively.
2. Increase corpus size.
3. Swap to your own corpus while preserving `id` and `text`.
4. Try the supplemental generation path.

## Supplemental generation run

If you want the same pipeline pattern for text generation, use:

```bash
sbatch jobs/run_generation_single_node.sh
```

Then validate:

```bash
python scripts/validate_outputs.py \
  --mode generation \
  --input-jsonl data/ag_news_generation_inputs.jsonl \
  --output-jsonl outputs/generation-baseline/generation_outputs.jsonl \
  --summary-json outputs/generation-baseline/run_summary.json
```

Expected generation output schema:

```json
{"id":"gen-0001","prompt":"Summarize this news item in one sentence:\n...","output_text":"..."}
```

Keep this supplemental. The main lesson is embeddings.

## Troubleshooting

- `GPU_VISIBLE_COUNT=0`: check the partition and runtime setup before debugging the model.
- output count or ID mismatch: rerun validation before using outputs downstream.
- inconsistent embedding dimensions or OOM: reduce `inference.batch_size` or sequence length before changing anything else.
- dataset download failure: fix data prep before debugging inference.

## Where this goes next

After a successful embeddings run, the natural next questions are:

- How do I build retrieval or RAG on top of these vectors?
- How do I evaluate output quality and throughput tradeoffs?
- When does it make sense to shard or scale inference further?

Those questions lead directly into the later lessons on RAG, evaluation, and topology-aware scaling.

## Navigation

- Previous extension lesson: [01. Foundation Model Adaptation](../01-foundation-model-adaptation/README.md)
- Next extension lesson: [03. RAG on MI250X](../03-rag-and-knowledge-workflows/README.md)
