# 01. Foundation Model Adaptation on LUMI-G

## Goal

Adapt a pretrained model to a small task on LUMI-G and produce a usable baseline checkpoint.

By the end of this lesson, you should be able to:

- explain what foundation-model adaptation means in this guide
- run a small adaptation workload on LUMI-G
- interpret the run outputs as adaptation signals, not just job artifacts
- choose a safe next modification after the baseline succeeds

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You already know how to run Python and submit a batch job on LUMI.
- `../../env.sh` is configured with a valid `CONTAINER`.

## Working directory

Run commands in this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/01-foundation-model-adaptation
```

## Adaptation in this lesson

This lesson uses:

- a pretrained text model: `distilbert-base-uncased`
- a task: 4-class news topic classification
- an input format: JSONL records with `text` and `label`

Unlike training from scratch, adaptation starts from pretrained weights and changes part or all of the model for a new task.

The key design choice is the adaptation mode:

- `head_only`: train only the classifier head
- `full`: update the whole model
- `lora`: keep the base model mostly fixed and train small adapter layers

The baseline uses `head_only` because it is the safest first run. `distilbert-base-uncased` and a small AG News subset keep the lesson focused on the adaptation pattern rather than task complexity.

## Minimal workflow

The core workflow has two steps:

1. prepare data
2. run adaptation

Load the lesson runtime in your shell:

```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch
source ../../env.sh
```

### Step 1: Prepare the sample dataset

Command:

```bash
python data/prepare_ag_news.py --output data/ag_news
```

This downloads a small AG News subset and writes `train.jsonl` and `eval.jsonl` in the format expected by the training config.

Example record:

```json
{"text":"Wall St. Bears Claw Back Into the Black","label":2}
```

### Step 2: Submit the baseline adaptation run

Command:

```bash
sbatch jobs/run_single_gcd.sh
```

This runs training inside one batch job.

Success signal in the Slurm output:

- `GPU_VISIBLE_COUNT=1` or greater
- training step logs appear
- `EVAL_LOSS=...`
- `EVAL_ACCURACY=...`
- `RUN_COMPLETE=1`

For this short baseline run, `jobs/run_single_gcd.sh` uses `dev-g`.

If you want a separate post-run check:

```bash
python scripts/validate_run.py --run-dir outputs/baseline-run
```

## How to interpret the result

Look for these signals:

- `GPU_VISIBLE_COUNT` confirms the runtime saw GPUs.
- training logs confirm the model, tokenizer, and dataset are wired correctly.
- `EVAL_LOSS` and `EVAL_ACCURACY` confirm the run produced measurable task output.
- `checkpoint/`, `metrics.json`, and `run_summary.json` confirm the result is reusable.

Expected layout:

- [assets/expected-output-tree.txt](assets/expected-output-tree.txt)

## What this successful baseline demonstrates

If the lesson works end to end, you have shown that:

- a pretrained model can be loaded in the intended LUMI runtime
- the task dataset format is valid for the adaptation script
- the model can be trained and evaluated on LUMI-G
- the run produces a checkpoint and machine-readable metrics

That is the lesson outcome. The commands are only the mechanism.

## What to change next

After the first successful run, change one thing at a time.

Recommended order:

1. Replace the sample JSONL files with your own dataset.
2. Adjust `training.batch_size` or `data.max_seq_len` conservatively.
3. Compare `adaptation.mode=head_only` with `adaptation.mode=lora`.
4. Use `jobs/run_single_node.sh` only after the single-GCD baseline is stable.

## Optional full-node visibility check

After the single-GCD baseline succeeds, you can inspect full-node visibility:

```bash
sbatch jobs/run_single_node.sh
```

Use this only after the single-GCD path is stable.

This is only a visibility check. It is not a real multi-GPU training run.

## Troubleshooting

- `GPU_VISIBLE_COUNT=0`: check the partition and runtime setup before debugging the model.
- `Set CONTAINER in env.sh` or container startup failure: fix `env.sh` first.
- dataset download, JSONL key errors, or OOM: rerun data prep, keep `text` and `label`, and stay with `head_only` before changing batch size or sequence length.

## Where this goes next

After you have a successful adapted checkpoint, the natural follow-on questions are:

- How do I run inference or embedding generation from this model?
- How do I evaluate whether this adaptation is actually good?
- When is it worth scaling beyond the single-device baseline?

Those questions lead directly into the later extension lessons on inference, evaluation, and topology-aware scaling.

## Navigation

- Previous core context: [5. Multi-GPU and Multi-Node Training](../../5-multi-gpu-and-node/README.md)
- Next extension lesson: [02. Inference and embeddings on MI250X](../02-inference-and-embeddings/README.md)
