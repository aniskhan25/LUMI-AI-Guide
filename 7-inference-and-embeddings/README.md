# 7. Inference and Embeddings

Run a HuggingFace model as a batch pipeline on LUMI — either to produce embeddings for search and retrieval, or to generate text outputs from a prompt corpus.

## Prerequisites

Check that `transformers` and `sentence-transformers` are available in the container:

```bash
singularity exec "$CONTAINER" python -c "import transformers, sentence_transformers; print('OK')"
```

If not, extend the container first — see the [container extension guide](https://github.com/aniskhan25/Extending-containers-on-LUMI/blob/main/README.org).

## Prepare the data

```bash
singularity exec "$CONTAINER" python data/prepare_ag_news.py --output data/ag_news
```

This writes:
- `data/ag_news/ag_news_corpus.jsonl` — 512 documents for embedding
- `data/ag_news/ag_news_generation_inputs.jsonl` — 32 prompt records for generation

## Embeddings

Embed a text corpus using a sentence embedding model. Each record in the output gets a dense vector alongside its original ID, ready for similarity search or as input to a RAG pipeline.

```bash
sbatch run_embeddings.sh
```

Output: `outputs/embeddings-baseline/embeddings.jsonl` — one record per document with `id` and `embedding` fields.

The default model is `sentence-transformers/all-MiniLM-L6-v2`. Change `model.name` in `configs/embeddings.yaml` to use a different model.

## Generation

Run batched text generation over a prompt corpus.

```bash
sbatch run_generation.sh
```

Output: `outputs/generation-baseline/generation_outputs.jsonl` — one record per prompt with `id`, `prompt`, and `output_text` fields.

The default model is `distilgpt2`. Change `model.name` in `configs/generation.yaml` to use a different causal LM.

## Bring your own data

Both scripts read JSONL input configured in the YAML file. For embeddings the expected fields are `id` and `text`; for generation, `id` and `prompt`. Point `data.input_jsonl` in the config to your own file.

## Next

You have completed the core guide. For advanced topics — RAG pipelines, evaluation, synthetic data, scaling patterns — see the [extension track](../extension-track/README.md).
