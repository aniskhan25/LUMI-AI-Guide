# Worked Example: Baseline Experiment

## Scenario

A team built a RAG workflow in previous lessons. Results are promising, but runs are difficult to compare because manifests and artifact paths are inconsistent.

## Baseline Artifact Snapshot

```text
runs/
  latest/
    output.jsonl
    notes.txt
  final/
    output.jsonl
```

Problems:

- ambiguous folder names (`latest`, `final`)
- no dataset or config version in outputs
- no promotion status

## Baseline Run Record (Missing Fields)

- run_id: present only in Slurm log file name
- dataset_version: missing
- config_version: missing
- model_or_adapter_ref: partial
- evaluation_summary: not linked

## Immediate Fix

Create `runs/draft/run-20260312-001/manifest.yaml` and move all outputs under that run directory.

Use [run-manifest.yaml](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/10-mlops-and-lifecycle-management/templates/run-manifest.yaml) as the canonical template.
