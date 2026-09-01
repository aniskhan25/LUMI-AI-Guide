# 1. Getting Started on LUMI

## Prerequisites

- A LUMI user account with GPU hours — run `lumi-workspaces` to check
- This repository cloned to `/project` or `/scratch`

```bash
git clone https://github.com/Lumi-supercomputer/LUMI-AI-Guide.git
cd LUMI-AI-Guide/1-quickstart
```

Update `PROJECT_ACCOUNT` in `../setup.sh` to match your project.

## The container

All lessons in this guide use the official AI container:

```
/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
```

Already set as the default in `../setup.sh`, which also loads the `Local-LAIF lumi-aif-singularity-bindings` module that gives the container access to the Slingshot network and your working directory.

The path is a pinned, date-stamped image rather than a `latest` symlink, so a new container release cannot change behaviour under you. Newer images appear in `/appl/local/laifs/containers/`; to use one, set `CONTAINER` in your environment or update `../setup.sh`. If you need extra packages, see the [container extension guide](https://github.com/aniskhan25/Extending-containers-on-LUMI/blob/main/README.org).

## Step 1: Smoke-test the container

```bash
sbatch run_base.sh
```

```bash
cat slurm-<jobid>.out
```

Look for `SMOKE TEST PASSED`.

## Step 2: Run the Vision Transformer

```bash
sbatch run_vit.sh
```

Uses `FakeData` — no dataset needed. When done, `vit_b_16_imagenet.pth` is written to this directory.

## Troubleshooting

- **Wrong account**: update `--account` in the job scripts or set `PROJECT_ACCOUNT` in `../setup.sh`
- **GPU not visible**: confirm `module load Local-LAIF lumi-aif-singularity-bindings` is in `../setup.sh` and that the job script sources it
- **Container not found**: confirm `CONTAINER` in `../setup.sh` points to a valid `.sif`

For general LUMI help, see the [LUMI documentation](https://docs.lumi-supercomputer.eu).

## Next

[2. Data on LUMI](../2-data/README.md)
