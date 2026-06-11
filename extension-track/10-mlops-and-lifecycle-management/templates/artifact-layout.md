# Artifact Layout Template

Use this structure to separate experiment and promoted assets.

```text
project-root/
  datasets/
    raw/
    prepared/
    versions/
  configs/
    experiments/
    promoted/
  runs/
    draft/
      run-<id>/
        manifest.yaml
        outputs/
        evaluation/
    reviewed/
      run-<id>/
        manifest.yaml
        review-notes.md
  promoted/
    version-<id>/
      manifest.yaml
      model-or-adapter/
      config/
      evaluation-summary/
      release-notes.md
  archive/
```

## Rules

- Never store promoted assets in draft run folders.
- Every run folder must include a manifest.
- Promoted version must reference exactly one source run ID.
- Evaluation summary must be colocated with promoted artifact.

