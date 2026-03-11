# 02. Efficient Inference and Embedding Pipelines on LUMI-G

This lesson is the next step after model adaptation: use a pretrained or adapted model as a scalable data-processing pipeline on LUMI-G.

## What This Lesson Enables

Run and validate a batch embedding workflow on LUMI-G, then optionally run a batched generation variant.

## When To Use This Workflow

Use this lesson when:

- you need corpus-scale model execution, not training
- throughput is more important than interactive latency
- outputs must be structured and reusable downstream

Use embeddings when you need:

- retrieval and semantic search
- clustering and deduplication
- downstream indexing and RAG building blocks

Use batched generation when you need:

- summaries or transformations
- structured synthetic text outputs

## Prerequisites

- Working LUMI access with GPU hours
- Familiarity with onboarding lessons and batch jobs
- `env.sh` configured with a valid `CONTAINER`
- Preferred: completion of Lesson 1 in this extension track

## Workflow At A Glance

```mermaid
flowchart LR
  A["Input JSONL corpus"] --> B["Batch loader"]
  B --> C["Model in AI Factory container"]
  C --> D["Embeddings output JSONL"]
  D --> E["Validation report"]
  E --> F["Downstream retrieval or analytics"]
```

## Minimal Working Example (Primary Path: Embeddings)

Work from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/02-inference-and-embeddings
```

1. Inspect or regenerate sample input:

```bash
python data/prepare_sample_data.py --output data
```

2. Run embeddings locally (dev path):

```bash
python scripts/run_embeddings.py --config configs/embeddings.yaml
```

3. Validate output completeness and schema:

```bash
python scripts/validate_outputs.py \
  --mode embeddings \
  --input-jsonl data/sample_corpus.jsonl \
  --output-jsonl outputs/embeddings-baseline/embeddings.jsonl \
  --summary-json outputs/embeddings-baseline/run_summary.json
```

4. Run the canonical Slurm job:

```bash
sbatch jobs/run_embeddings_single_node.sh
```

## Optional Secondary Path (Generation)

Run batched generation:

```bash
python scripts/run_generation.py --config configs/generation.yaml
```

Validate:

```bash
python scripts/validate_outputs.py \
  --mode generation \
  --input-jsonl data/sample_generation_inputs.jsonl \
  --output-jsonl outputs/generation-baseline/generation_outputs.jsonl \
  --summary-json outputs/generation-baseline/run_summary.json
```

Or use Slurm:

```bash
sbatch jobs/run_generation_single_node.sh
```

## How To Verify It Worked

Check all of these:

- log line `GPU_VISIBLE_COUNT=<n>`
- output JSONL file exists and is non-empty
- output record count matches input count
- all input IDs appear in output
- embedding dimensions are consistent (primary path)
- run summary JSON exists and reports completion

Expected output layout: [assets/expected-output-tree.txt](assets/expected-output-tree.txt)  
Expected record schemas: [assets/expected-output-schema.md](assets/expected-output-schema.md)

## Throughput Thinking On LUMI-G

- Batch processing is the default pattern for corpus workloads.
- Throughput depends heavily on batch size, max sequence length, and model size.
- One-sample-at-a-time prompt processing is usually a poor HPC pattern.
- Inference still has memory limits; tune batch size before scaling out.

## LUMI-G Details That Matter For Inference

- Confirm GPU visibility explicitly; inference can silently run on CPU if not checked.
- Memory pressure still matters even without backpropagation.
- Start from a simple, reproducible baseline and scale with measured changes.
- Do not assume larger GPU counts always improve end-to-end throughput for small batches.

## Common Failure Modes

See [troubleshooting/common-failures.md](troubleshooting/common-failures.md).

## How To Extend

After baseline success:

- increase `inference.batch_size` incrementally
- increase corpus size and validate output integrity again
- switch output path to project-specific storage
- shard the corpus across multiple batch jobs
- move from embeddings to generation/classification using the same pipeline pattern

## Operational Checklist

- Input file schema validated (`id`, `text` or `prompt`)
- Output path writable
- Container selected (`CONTAINER` in `env.sh`)
- GPU visibility confirmed in logs
- Batch size chosen for memory constraints
- IDs preserved in outputs
- Output count matches input count
- Summary file and validation result saved

## Next Lesson

Natural follow-on options:

- retrieval and RAG workflow construction on LUMI-G
- topology-aware scaling for high-throughput AI workloads

