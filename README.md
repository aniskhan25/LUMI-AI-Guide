# LUMI AI Guide

> **Disclaimer:** This is not the official LUMI AI guide. For the official guide, see [github.com/Lumi-supercomputer/LUMI-AI-Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide).

A practical guide for running AI training on LUMI, built around a runnable Vision Transformer example in PyTorch.

The official container for all lessons:

```
/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
```

This is a pinned, date-stamped image rather than a `latest` symlink, so a new container
release cannot change behaviour under you. Newer images appear in
`/appl/local/laifs/containers/`; to use one, set `CONTAINER` in your environment or update
`setup.sh`.

## Core lessons

- [1. Getting Started on LUMI](1-quickstart/README.md)
- [2. Data on LUMI](2-data/README.md)
- [3. Multi-GPU and Multi-Node Training](3-multi-gpu-and-node/README.md)
- [4. Monitoring and Profiling](4-monitoring-and-profiling/README.md)
- [5. Experiment Tracking](5-experiment-tracking/README.md)
- [6. Foundation Model Adaptation](6-foundation-model-adaptation/README.md)
- [7. Inference and Embeddings](7-inference-and-embeddings/README.md)

## Before you start

- You need a LUMI user account and a project with GPU hours — run `lumi-workspaces` to check
- Update `PROJECT_ACCOUNT` in `setup.sh` before submitting any jobs
- Clone this repository to `/project` or `/scratch`, not `$HOME`

## Further reading

- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)
- [Data-Aware AI on LUMI](https://github.com/aniskhan25/data-aware-ai)
- [Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai)
- [Container extension guide](https://github.com/aniskhan25/Extending-containers-on-LUMI/blob/main/README.org)
- [LUMI AI workshop materials](https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20240529/)
