# 01. Foundation Model Adaptation on MI250X

After the onboarding guide taught how to run and scale a baseline ML workflow on LUMI, this lesson introduces the first advanced AI Factory pattern: adapting a pretrained foundation model on LUMI-G with the supported container environment.

## What This Lesson Enables

Run and verify a small but real foundation-model adaptation workflow on LUMI-G, then make one controlled modification safely.

## When To Use This Workflow

Use this lesson when:

- you already have LUMI onboarding basics working
- you need a first production-like adaptation path
- you want a container-first pattern that can scale later

Do not use this lesson as a replacement for:

- account setup and Slurm basics
- full multi-node distributed training design
- inference serving and RAG system architecture

## Prerequisites

- Working LUMI access and project GPU hours
- Baseline familiarity with shell, Python, and training loops
- `env.sh` configured with a valid `CONTAINER`

## Workflow At A Glance

```mermaid
flowchart LR
  A["JSONL data"] --> B["AI Factory container"]
  B --> C["Pretrained text model"]
  C --> D["Adaptation run on LUMI-G"]
  D --> E["Checkpoint + metrics + logs"]
  E --> F["Validation checklist"]
```

## Minimal Working Example

Work from this directory:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/01-foundation-model-adaptation
```

1. Prepare sample dataset:

```bash
python data/prepare_sample_data.py --output data/sample_data
```

2. Run one adaptation pass (local/dev):

```bash
python scripts/train.py --config configs/baseline.yaml
```

3. Validate outputs:

```bash
python scripts/validate_run.py --run-dir outputs/baseline-run
```

4. Run on LUMI with Slurm:

```bash
sbatch jobs/run_single_gcd.sh
```

## Verification Signals

Look for all of these:

- `GPU_VISIBLE_COUNT=<n>` in logs
- training step logs with decreasing/steady loss behavior
- output run directory with checkpoint files
- `metrics.json` and `run_summary.json`
- successful `validate_run.py` output

Expected layout is shown in [assets/expected-output-tree.txt](assets/expected-output-tree.txt).

## LUMI-G Details That Matter

- LUMI-G uses AMD MI250X nodes.
- Software typically sees 8 GPU/GCD devices per node.
- CPU/GPU locality matters; avoid making topology assumptions when scaling.

This lesson keeps the baseline path simple, then points to scale-up next steps.

## Common Failure Modes

See [troubleshooting/common-failures.md](troubleshooting/common-failures.md).

## How To Extend

After baseline success:

- increase batch size conservatively
- run `jobs/run_single_node.sh` for full-node visibility checks
- replace sample dataset with your own JSONL dataset
- switch adaptation mode from `head_only` to `lora` after confirming `peft` availability

## Operational Checklist

- Container selected (`CONTAINER` in `env.sh`)
- Binding module loaded in Slurm script
- Data path exists and sample records are valid JSONL
- Output path is writable
- GPU visibility confirmed in logs
- Checkpoint and metrics files created

## Next Lesson

Natural follow-on options:

- topology-aware scaling for adaptation workloads
- efficient inference and embedding pipelines

