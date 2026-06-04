# 1. Getting Started on LUMI

## Prerequisites

- A LUMI user account with GPU hours — run `lumi-workspaces` to check
- This repository cloned to `/project` or `/scratch`

```bash
git clone https://github.com/Lumi-supercomputer/LUMI-AI-Guide.git
cd LUMI-AI-Guide/1-quickstart
```

Update `PROJECT_ACCOUNT` and `LUMI_USER` in `../env.sh` to match your project.

## The container

All lessons in this guide use the official AI container:

```
/appl/local/laifs/containers/lumi-multitorch-latest.sif
```

Already set as the default in `../env.sh`. If you need extra packages, see the [container extension guide](https://github.com/aniskhan25/Extending-containers-on-LUMI/blob/main/README.org).

## Step 1: Smoke-test the container

```bash
sbatch run_base.sh
cat slurm-<jobid>.out
```

Look for `SMOKE TEST PASSED`.

## Step 2: Run the Vision Transformer

```bash
sbatch run_vit.sh
```

Uses `FakeData` — no dataset needed. When done, `vit_b_16_imagenet.pth` is written to this directory.

## Troubleshooting

- **Wrong account**: update `--account` in the job scripts or set `PROJECT_ACCOUNT` in `../env.sh`
- **GPU not visible**: confirm `module load singularity-AI-bindings` is in the job script
- **Container not found**: confirm `CONTAINER` in `../env.sh` points to a valid `.sif`

For general LUMI help, see the [LUMI documentation](https://docs.lumi-supercomputer.eu).

## Next

[2. Data: formats and storage](../3-file-formats/README.md)
