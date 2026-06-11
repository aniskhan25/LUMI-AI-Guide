# 6. Foundation Model Adaptation

Fine-tune a pretrained model on LUMI for a downstream task. Use adaptation when prompting alone is too unstable or the desired behaviour should live in the checkpoint itself.

## Prerequisites

Source the environment first, then check that `transformers` and `datasets` are available in the container:

```bash
source ../setup.sh
singularity exec "$CONTAINER" python -c "import transformers, datasets; print('OK')"
```

If not, extend the container first — see the [container extension guide](https://github.com/aniskhan25/Extending-containers-on-LUMI/blob/main/README.org).

## Prepare the data

Run the data preparation step as a batch job — downloading and processing data on the login node is not allowed on LUMI:

```bash
sbatch prepare_data.sh
```

This writes `data/ag_news/train.jsonl` and `data/ag_news/eval.jsonl` (AG News, 4-class news topic classification).

## Adaptation modes

Edit `configs/baseline.yaml` to choose the adaptation mode:

| Mode | What trains | When to use |
|---|---|---|
| `head_only` | Classifier head only | Start here — fastest, lowest risk |
| `lora` | Small adapter layers (requires `peft`) | When head_only is not enough |
| `full` | All parameters | Maximum flexibility, highest cost |

## Run

Single GCD:

```bash
sbatch run_finetune.sh
```

Scale to a full node (8 GCDs) once the single-GCD baseline is stable:

```bash
sbatch run_finetune_ddp.sh
```

On success you will see:

```
EVAL_LOSS=...
EVAL_ACCURACY=...
RUN_COMPLETE=1
```

The checkpoint is saved to `outputs/baseline-run/checkpoint/`.

## Bring your own data

Replace `data/prepare_ag_news.py` with your own data preparation script. The training scripts expect JSONL with `text` (string) and `label` (int) fields. Update `num_labels` in `configs/baseline.yaml` to match your class count.

## Troubleshooting

- **`import transformers` fails**: extend the container as described above
- **CUDA out of memory**: reduce `batch_size` in `configs/baseline.yaml`, or switch to `lora`
- **Poor accuracy with `head_only`**: try `lora` before increasing batch size or epochs

## Next

A natural next step is running the adapted model at scale — for batch inference or to build embeddings for search and RAG:

- [7. Inference and Embeddings](../7-inference-and-embeddings/README.md) — batch-embed a corpus or generate outputs with a HuggingFace model
