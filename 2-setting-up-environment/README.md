# 2. Setting up your own environment

## Goal

Define one stable runtime baseline that is reused by chapters `3` to `9`.

## Assumptions

- You completed [1. QuickStart](../1-quickstart/README.md).
- You can run Slurm jobs on LUMI.
- You are working from a clone of this repository on `/project` or `/scratch`.

## What changes from baseline

- Baseline you already have: a working single-GPU run from QuickStart.
- This lesson adds: one reproducible container workflow and environment-extension pattern for the rest of the guide.
- Expected output/artifact: a validated container + Python environment contract used consistently in later lessons.

## Minimal run checkpoint

Allocate an interactive GPU node:

```bash
salloc --account=project_462000131 --partition=small-g \
  --nodes=1 --gpus-per-node=1 --ntasks=1 --cpus-per-task=7 \
  --mem-per-gpu=60G --time=00:15:00
```

Then run:

```bash
source ../env.sh
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
srun singularity exec "$CONTAINER" python -c "import torch; print(torch.cuda.device_count())"
```

Success signal:

- Command prints a GPU count greater than `0`.

## Baseline contract for chapters 3-9

Use these conventions in later lessons unless explicitly overridden:

- Runtime launcher: `srun singularity exec ...` (not `singularity run`).
- Container selection: `CONTAINER` from `../env.sh`.
- Container bindings: `module load singularity-AI-bindings`.
- Python environment extension: QuickStart squashfs by default; optional local venv only for custom experiments.
- Data path variables: use paths exported in `env.sh` (for example `TINY_HDF5_PATH`, `TINY_LMDB_PATH`, `SQUASH_LARGE`).

## Recommended baseline workflow

### 1. Load runtime bindings and shared environment

```bash
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
source ../env.sh
```

### 2. Validate container + GPU access

```bash
srun singularity exec "$CONTAINER" python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 3. Inspect installed Python packages (optional)

```bash
srun singularity exec "$CONTAINER" pip list | head -n 30
```

### 4. Extend the Python environment

For this guide, the standard extension path is the squashfs build from QuickStart:

```bash
cd ../1-quickstart
./build_visiontransformer_sqsh.sh
```

This produces `../resources/visiontransformer-env.sqsh`, reused by later run scripts.

### 5. Environment extension options for later chapters

Use this decision rule in chapters `3` to `9`:

- Recommended default: reuse `../resources/visiontransformer-env.sqsh`.
- Optional fallback: create a local `venv` only for custom experiments.

If you need an optional local `venv` (run from lesson `2-setting-up-environment`):

```bash
cd ../2-setting-up-environment
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
source ../env.sh

# choose any working directory where you want to keep the venv
mkdir -p ./optional-venv-workdir
cd ./optional-venv-workdir

singularity exec "$CONTAINER" bash -c '
python -m venv .venv --system-site-packages
.venv/bin/python -m pip install h5py lmdb msgpack six tqdm mlflow
'
```

Notes:

- Chapter `3` scripts are validated against the sqsh path, not this optional venv path.
- Keep optional venv usage limited to custom runs where you directly call `.venv/bin/python`.

## Why containers on LUMI

Containers on LUMI are used for:

- ROCm and Slingshot compatibility
- reduced filesystem pressure
- reproducible framework/runtime combinations

These containers are tuned for LUMI and are generally not portable as-is to other systems.

## Optional local venv reference

If you choose the optional venv route, run a lightweight GPU check with that interpreter:

```bash
cd ./optional-venv-workdir
singularity exec "$CONTAINER" bash -c '.venv/bin/python -c "import torch; print(torch.cuda.device_count())"'
```

Expected output is a value greater than `0`.

## Optional alternatives (advanced/reference)

- EasyBuild wrapper modules can be used instead of direct `singularity exec`.
- Custom images are possible, but start from LUMI-provided base images to keep ROCm/Slingshot compatibility.
- If you manage all bindings manually, keep `singularity-AI-bindings` behavior as your reference baseline.

## Used by next lessons

- [3. File formats for training data](../3-file-formats/README.md): uses same container execution + data-path conventions.
- [4. Data Storage Options](../4-data-storage/README.md): assumes same runtime while changing data placement.
- [5. Multi-GPU and Multi-Node Training](../5-multi-gpu-and-node/README.md): builds on this baseline for DDP/DeepSpeed launchers.
- [7-9 tracking lessons](../7-tensorboard-visualization/README.md): reuse the same runtime, adding logging integrations.

## Verify

Confirm the following before moving on:

- You can run Python in the container with GPU visibility.
- `env.sh` values are resolved correctly in your job or interactive session.
- You can execute guide scripts with the base container + squashfs extension.

## Troubleshooting

- `Set CONTAINER in env.sh`: define `CONTAINER` in `env.sh` and ensure the file exists.
- GPU count is zero: load `singularity-AI-bindings` and run inside an allocated GPU job/step.
- Import errors for extra packages: confirm your runtime extension method (sqsh by default, optional local venv if you chose it).

## Navigation

- Previous: [1. QuickStart](../1-quickstart/README.md)
- Next: [3. File formats for training data](../3-file-formats/README.md)
