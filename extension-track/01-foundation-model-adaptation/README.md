# 01. Foundation Model Adaptation on MI250X

## Goal

Reuse the same LUMI job pattern from the core guide, but apply it to one real foundation-model adaptation workflow.

By the end of this lesson, you should be able to:

- prepare a small JSONL text dataset
- adapt a pretrained text model on LUMI-G
- verify GPU visibility, logs, and output artifacts
- make one safe change to the baseline run

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You are comfortable with the single-device baseline before attempting scale-up.
- `../../env.sh` is configured with a valid `CONTAINER`.

## Working directory

Run commands in this chapter from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/01-foundation-model-adaptation
```

## What changes from baseline

- Baseline you already have: containerized jobs on LUMI, GPU visibility checks, Slurm submission, and artifact verification.
- This lesson adds: adaptation of a pretrained text model instead of training the Vision Transformer example from scratch.
- Expected output/artifact: a run directory with a saved checkpoint, `metrics.json`, and `run_summary.json`.

## What stays the same from earlier lessons

- Runtime launcher: same LUMI container workflow and `env.sh` contract.
- Validation habit: check logs first, then confirm expected files.
- Progression: start with the smallest working path, then extend carefully.

If the earlier guide taught you how to run one stable GPU job on LUMI, this lesson teaches you how to reuse that pattern for a foundation-model use case.

## What this lesson introduces

This lesson uses a pretrained NLP model (`distilbert-base-uncased`) for binary text classification.

The important new idea is adaptation:

- You are not training a foundation model from scratch.
- You start from pretrained weights.
- You change either only the task head (`head_only`), the full model (`full`), or later a parameter-efficient path such as `lora`.

The default baseline is `head_only`, because it is the safest first run.

## How this chapter reaches the goal

1. Generate a small JSONL train/eval dataset.
2. Run a short adaptation job with the baseline config.
3. Validate the run directory and GPU visibility record.
4. Repeat the same workflow through Slurm on LUMI-G.
5. Make one controlled modification only after the baseline succeeds.

## Minimal run checkpoint

From the lesson directory, run these commands in order:

1. Prepare the sample dataset:

```bash
python data/prepare_sample_data.py --output data/sample_data
```

2. Run one short adaptation pass:

```bash
python scripts/train.py --config configs/baseline.yaml
```

3. Validate outputs:

```bash
python scripts/validate_run.py --run-dir outputs/baseline-run
```

Success signal:

- Training prints `GPU_VISIBLE_COUNT=<n>`.
- The run completes and prints `RUN_COMPLETE=1`.
- Validation prints `VALIDATION_OK=1`.

Note:

- The default config requires GPU visibility.
- For non-LUMI local debugging only, you may temporarily enable CPU fallback in [configs/baseline.yaml](configs/baseline.yaml) or use `--allow-cpu` with the validator.

## Minimal LUMI run checkpoint

Once the local baseline path is understood, submit the LUMI job:

```bash
sbatch jobs/run_single_gcd.sh
```

Success signal:

- Job output shows `GPU_VISIBLE_COUNT=1` or greater.
- Training logs appear without container or CUDA visibility errors.
- Output directory contains the expected files listed below.

After that, you can optionally inspect full-node visibility:

```bash
sbatch jobs/run_single_node.sh
```

## Baseline contract for this chapter

- Use `CONTAINER` from `../../env.sh`.
- Keep the first run in `adaptation.mode=head_only`.
- Use the provided JSONL schema: `text` and `label`.
- Validate artifacts before changing batch size, model, or adaptation mode.
- Treat single-device success as the prerequisite for any scale-up decision.

## Files that matter

- Training entrypoint: [scripts/train.py](scripts/train.py)
- Baseline config: [configs/baseline.yaml](configs/baseline.yaml)
- Sample data generator: [data/prepare_sample_data.py](data/prepare_sample_data.py)
- Validation script: [scripts/validate_run.py](scripts/validate_run.py)
- Single-GCD jobscript: [jobs/run_single_gcd.sh](jobs/run_single_gcd.sh)

## Verification

Confirm all of the following:

- Logs include `GPU_VISIBLE_COUNT=<n>`.
- Training step logs appear during the run.
- The output run directory exists.
- `checkpoint/` contains saved model/tokenizer files.
- `metrics.json` exists.
- `run_summary.json` exists.
- Validation completes successfully.

Expected layout:

- [assets/expected-output-tree.txt](assets/expected-output-tree.txt)

## Why this works on LUMI-G

- The lesson uses the same container-first execution pattern as the earlier guide.
- LUMI-G MI250X nodes expose GPU/GCD devices to the runtime inside the container.
- The first baseline stays deliberately small so you can validate the workflow before making topology or scale assumptions.

## Recommended first extension after baseline

Change only one thing at a time:

1. Increase `training.batch_size` conservatively.
2. Replace the sample JSONL files with your own dataset.
3. Try `adaptation.mode=lora` only after confirming `peft` is available in your runtime.
4. Use `jobs/run_single_node.sh` before discussing multi-node adaptation.

## Troubleshooting

See [troubleshooting/common-failures.md](troubleshooting/common-failures.md).

The most common issues are:

- GPU not visible inside the container
- missing or invalid `CONTAINER` path
- JSONL schema mismatches
- out-of-memory from changing batch size or sequence length too early

## Navigation

- Previous core context: [5. Multi-GPU and Multi-Node Training](../../5-multi-gpu-and-node/README.md)
- Next extension lesson: [02. Inference and embeddings on MI250X](../02-inference-and-embeddings/README.md)
