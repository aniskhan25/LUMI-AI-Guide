# 6. Foundation Model Adaptation

Fine-tune a pretrained model on LUMI for a downstream task. This is the right approach when you need the model to learn a specific task boundary or label mapping that prompting alone cannot reliably produce.

## Prerequisites

This lesson requires `transformers` and `datasets` from HuggingFace. Check whether they are available in the container:

```bash
singularity exec "$CONTAINER" python -c "import transformers, datasets; print('OK')"
```

If not, extend the container first — see the [container extension guide](https://github.com/aniskhan25/Extending-containers-on-LUMI/blob/main/README.org).

## Adaptation modes

Three modes are available in `finetune.py`:

| Mode | What trains | When to use |
|---|---|---|
| `head_only` | Classifier head only | First run — fastest, lowest risk |
| `lora` | Small adapter layers (requires `peft`) | When head_only is not enough |
| `full` | All parameters | Maximum flexibility, highest cost |

Start with `head_only`. Move to `lora` or `full` only after the baseline succeeds.

## Run

```bash
sbatch run_finetune.sh             # head_only (default)
sbatch run_finetune.sh lora        # lora adapters
sbatch run_finetune.sh full        # full fine-tuning
```

The script fine-tunes `distilbert-base-uncased` on a subset of AG News (4-class news topic classification). On success you will see:

```
Eval accuracy: 0.xxxx
RUN_COMPLETE=1
Checkpoint saved to outputs/finetune-head_only/checkpoint
```

Metrics are written to `outputs/finetune-<mode>/metrics.json`.

## Bring your own data

Replace the `load_dataset("ag_news")` call in `finetune.py` with your own dataset. The script expects samples with `text` (string) and `label` (int) fields. Adjust `num_labels` in the model initialisation to match your class count.

## Scaling to multiple GPUs

Once the single-GPU baseline is stable, the same script can be wrapped with `torch.distributed.run` using the DDP patterns from [L3](../3-multi-gpu-and-node/README.md). Scale only after the single-GPU run is working correctly.

## Troubleshooting

- **`import transformers` fails**: extend the container as described above
- **CUDA out of memory**: reduce `--batch_size` or `--max_len`, or switch to `lora`
- **Poor accuracy with `head_only`**: try `lora` before increasing batch size or epochs

## Next

You have completed the core guide.

A natural next step is running the fine-tuned model at scale — for batch inference or to build embeddings for search and RAG. The extension track starts there:

- [EXT-02: Inference and embeddings on MI250X](../extension-track/02-inference-and-embeddings/README.md) — batch-embed a corpus or generate outputs with a HuggingFace model
- [EXT-03: RAG on MI250X](../extension-track/03-rag-and-knowledge-workflows/README.md) — chunk, embed, index, and query a document corpus

For evaluation, synthetic data, and advanced serving patterns see the full [extension track index](../extension-track/README.md).
