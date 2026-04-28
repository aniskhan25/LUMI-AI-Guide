# 01. Foundation Model Adaptation on LUMI-G

## Goal

This lesson introduces the first AI Factory workflow in the extension track: take a pretrained model, adapt it to a small task on LUMI-G, and verify that the result is a usable baseline rather than just a successful batch job.

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

## What is new in this lesson

Earlier lessons established the operational baseline: how to run code on LUMI, how to use the provided software environment, and how to inspect logs and artifacts.

This lesson adds one new capability: adapting a pretrained model for a concrete downstream task.

In this lesson, adaptation means:

- the model does not start from random weights
- the pretrained model already knows useful language patterns
- you change part or all of that pretrained model so it performs a new task

That is different from training from scratch, where the model begins with random parameters and must learn everything from the task data alone.

## What foundation-model adaptation means here

This lesson uses:

- a pretrained text model: `distilbert-base-uncased`
- a task: binary text classification
- an input format: JSONL records with `text` and `label`

The important design choice is the adaptation mode:

- `head_only`: freeze most of the pretrained model and train only the classifier head
- `full`: update the whole model
- `lora`: keep the base model mostly fixed and train small adapter layers

The baseline starts with `head_only` because it is the lowest-risk way to answer the first question:

Can I run a valid pretrained-model adaptation workflow on LUMI-G and produce a checkpoint, metrics, and a reproducible output directory?

## Why this baseline was chosen

This lesson is deliberately small.

- `distilbert-base-uncased` is a practical starter model because it is widely supported and light enough for a first adaptation run.
- Binary text classification is simple enough to keep the focus on the adaptation workflow, not on task complexity.
- The sample JSONL dataset is synthetic on purpose. It gives you a controlled baseline before you swap in real data.
- `head_only` reduces the amount of trainable state, which lowers both runtime risk and interpretation noise for the first run.

The lesson is not trying to prove that this is the best possible model or task. It is trying to teach a reusable adaptation pattern on LUMI.

## Minimal workflow

The core workflow has three steps:

1. prepare data
2. run adaptation
3. validate outputs

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
python data/prepare_sample_data.py --output data/sample_data
```

What this does:

- writes `train.jsonl` and `eval.jsonl`
- creates the minimal text-classification dataset expected by the training config

Why it matters:

- adaptation still needs task-specific data
- this step makes the dataset contract explicit: one JSON object per line with `text` and `label`

Success signal:

- `data/sample_data/train.jsonl` exists
- `data/sample_data/eval.jsonl` exists

Example record:

```json
{"text":"training job completed with stable loss","label":1}
```

### Step 2: Submit the baseline adaptation run

Command:

```bash
sbatch jobs/run_single_gcd.sh
```

What this does:

- starts a short single-GCD baseline run on LUMI-G
- loads the AI Factory container environment from `env.sh`
- runs data preparation, training, and validation inside the batch job

Why it matters:

- this is the first lesson where the workload is a pretrained-model adaptation run rather than a generic training smoke test
- success here means the full adaptation path works end to end on LUMI-G

Success signal in the Slurm output:

- `GPU_VISIBLE_COUNT=1` or greater
- training step logs appear
- `EVAL_LOSS=...`
- `EVAL_ACCURACY=...`
- `RUN_COMPLETE=1`
- `VALIDATION_OK=1`

For this short baseline run, `jobs/run_single_gcd.sh` uses `dev-g`.

### Step 3: Re-check the run directory manually

The batch job already runs validation internally, but it is useful to inspect the run again yourself.

Command:

```bash
python scripts/validate_run.py \
  --run-dir "${SCRATCH_ROOT}/foundation-adaptation/baseline-run" \
  --min-accuracy 0.0
```

What this does:

- checks that the run directory exists
- confirms checkpoint and metrics files are present
- confirms GPU visibility was recorded in the run summary

Why it matters:

- adaptation on LUMI is not only about whether the job finished
- it is also about whether the job produced artifacts you can use in later lessons

## How to interpret the result

Do not stop at “the job passed.”

Read the outputs in this order:

### 1) `GPU_VISIBLE_COUNT`

What it tells you:

- whether the training script saw GPU devices in the intended runtime

What a good baseline means:

- the software environment and batch launch were compatible with the adaptation workload

### 2) Training loss logs

What they tell you:

- whether optimization is progressing at all

What a good baseline means:

- the model, tokenizer, dataset, and loss path are wired together correctly

You are not looking for perfect convergence here. You are looking for a stable, believable adaptation run.

### 3) `EVAL_LOSS` and `EVAL_ACCURACY`

What they tell you:

- whether the adapted model can run evaluation on held-out data

What a good baseline means:

- the run produced measurable task output, not just a checkpoint file

With the synthetic sample dataset, these metrics are smoke-test indicators, not final model-quality claims.

### 4) Checkpoint contents

Expected layout:

- [assets/expected-output-tree.txt](assets/expected-output-tree.txt)

What it tells you:

- whether the adapted model and tokenizer were saved in a reusable format

What a good baseline means:

- later lessons can consume these outputs for inference, evaluation, or further experimentation

### 5) `metrics.json` and `run_summary.json`

What they tell you:

- the metrics file captures outcome numbers
- the summary file captures run identity, device, and output location

What a good baseline means:

- the run is inspectable and reproducible enough to compare against later modifications

## Files that matter

- Training entrypoint: [scripts/train.py](scripts/train.py)
- Baseline config: [configs/baseline.yaml](configs/baseline.yaml)
- Sample data generator: [data/prepare_sample_data.py](data/prepare_sample_data.py)
- Validation script: [scripts/validate_run.py](scripts/validate_run.py)
- Single-GCD jobscript: [jobs/run_single_gcd.sh](jobs/run_single_gcd.sh)

## What this successful baseline demonstrates

If the lesson works end to end, you have shown that:

- a pretrained model can be loaded in the intended LUMI runtime
- the task dataset format is valid for the adaptation script
- the model can be trained and evaluated on LUMI-G
- the run produces a checkpoint and machine-readable metrics

That is the real lesson outcome. The commands are only the mechanism.

## What to change next

After the first successful run, change one thing at a time.

Recommended order:

1. Replace the sample JSONL files with your own dataset.
2. Adjust `training.batch_size` or `data.max_seq_len` conservatively.
3. Compare `adaptation.mode=head_only` with `adaptation.mode=lora`.
4. Use `jobs/run_single_node.sh` only after the single-GCD baseline is stable.

Why this order:

- changing data first tests whether your real task fits the lesson pattern
- changing batch size or sequence length tests resource behavior
- trying LoRA tests a more realistic parameter-efficient adaptation path
- scaling before the single-device path is stable makes failures harder to interpret

## Optional full-node visibility check

After the single-GCD baseline succeeds, you can inspect full-node visibility:

```bash
sbatch jobs/run_single_node.sh
```

This is not the main lesson goal. Use it only after the baseline path is already trustworthy.

## Troubleshooting

### 1) No GPU visible inside container

Symptoms:

- `GPU_VISIBLE_COUNT=0`
- train script exits with a CUDA visibility error

Checks:

- Ensure you loaded the CSC module path and PyTorch module:
  - `module use /appl/local/csc/modulefiles`
  - `module load pytorch`
- Confirm the batch job runs on a GPU partition:
  - `dev-g` for the short baseline run
  - `standard-g` for the full-node run

### 2) Missing or wrong container path

Symptoms:

- `Set CONTAINER in env.sh`
- `singularity exec` fails before Python starts

Checks:

- Set a valid `CONTAINER` in [env.sh](/Users/anisrahm/Documents/LUMI-AI-Guide/env.sh)
- Verify that the container path exists and is readable on LUMI

### 3) JSONL parse or key errors

Symptoms:

- `KeyError: text`
- `KeyError: label`
- JSON decode failures

Checks:

- Rebuild the sample data:

```bash
python data/prepare_sample_data.py --output data/sample_data
```

- Ensure each JSONL record contains exactly the expected keys:
  - `text`
  - `label`

### 4) Out-of-memory

Symptoms:

- runtime OOM
- sudden process termination during the forward pass

Checks:

- Reduce `training.batch_size`
- Reduce `data.max_seq_len`
- Keep `adaptation.mode=head_only` for the first baseline run

### 5) Poor scaling assumptions

Symptoms:

- full-node runs are slower than expected

Checks:

- Do not assume simple CPU/GPU numbering alignment on MI250X/GCD topology
- Start from the single-device baseline and profile before making scaling decisions

## Where this goes next

After you have a successful adapted checkpoint, the natural follow-on questions are:

- How do I run inference or embedding generation from this model?
- How do I evaluate whether this adaptation is actually good?
- When is it worth scaling beyond the single-device baseline?

Those questions lead directly into the later extension lessons on inference, evaluation, and topology-aware scaling.

## Navigation

- Previous core context: [5. Multi-GPU and Multi-Node Training](../../5-multi-gpu-and-node/README.md)
- Next extension lesson: [02. Inference and embeddings on MI250X](../02-inference-and-embeddings/README.md)
