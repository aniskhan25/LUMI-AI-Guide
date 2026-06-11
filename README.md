# LUMI AI Guide

A practical guide for running AI training on LUMI, built around a runnable Vision Transformer example in PyTorch.

The official container for all lessons:

```
/appl/local/laifs/containers/lumi-multitorch-latest.sif
```

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
- Update `PROJECT_ACCOUNT` and `LUMI_USER` in `env.sh` before submitting any jobs
- Clone this repository to `/project` or `/scratch`, not `$HOME`

## Further reading

- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)
- [Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai)
- [Container extension guide](https://github.com/aniskhan25/Extending-containers-on-LUMI/blob/main/README.org)
- [LUMI AI workshop materials](https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20240529/)
