# 3. File formats for training data

## Goal

Choose a practical training-data format on LUMI and run one validated loader benchmark.

## Assumptions

- You completed [2. Setting up your own environment](../2-setting-up-environment/README.md).
- You can run containerized jobs with `srun singularity exec`.
- You have dataset paths configured via `env.sh`.

## What changes from baseline

- Baseline you already have: a stable runtime environment and container workflow.
- This lesson adds: data-format selection and conversion strategy for SquashFS, HDF5, and LMDB.
- Expected output/artifact: one chosen format, converted data artifact(s), and a successful benchmark run.

## Minimal run checkpoint

From a clean state, run these commands one by one.

1. Download tiny-ImageNet data:

```bash
bash ./get_data.sh
```

2. Convert to LMDB:

```bash
sbatch convert.sh lmdb
```

3. Wait for the conversion job to finish successfully, then run benchmark:

```bash
sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh lmdb
```

Success signal:

- Job output includes a `dataloader time:` line (for example `LMDB dataloader time:`).

## Baseline contract for this chapter

- Submit jobs from `3-file-formats/` so relative paths resolve.
- Use data-path variables from `../env.sh` (`TINY_HDF5_PATH`, `TINY_LMDB_PATH`, `SQUASH_LARGE`, etc.).
- Runtime extension policy: scripts use `../resources/visiontransformer-env.sqsh`.
- Avoid unpacking large archives into raw file trees on Lustre when possible.
- For environment customization options, follow [2. Setting up your own environment](../2-setting-up-environment/README.md).

## Recommended workflow

1. Download/stage raw input data.
2. Pick a format using the quick guide below.
3. Convert data once.
4. Run a small benchmark/smoke test.
5. Keep benchmark results and proceed with the chosen format.

### Tiny dataset setup used in this chapter

1. Download raw tiny-ImageNet data:

```bash
bash ./get_data.sh
```

2. Convert to a target format:

```bash
sbatch convert.sh lmdb
```

3. After conversion finishes, run the benchmark:

```bash
sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh lmdb
```

## Quick format guide

Use the script workflow above for this guide.  
The format sections below are background/reference plus pointers to the exact repo scripts.

| Format | Best fit | Main tradeoff | Chapter scripts |
| :-- | :-- | :-- | :-- |
| SquashFS | Fastest setup, minimal custom code, strong filesystem behavior | Slower on very large workloads than LMDB | `scripts/squashfs/*` |
| HDF5 | Regular/fixed-shape data and simple numpy-style indexing | Poor fit for variable-size large image corpora | `scripts/hdf5/*` |
| LMDB | Large/variable-size image datasets and high read performance | More custom conversion/dataset logic | `scripts/lmdb/*` |

## SquashFS

In this repo, use:

```bash
sbatch convert.sh squashfs
```

After conversion finishes, run benchmark:

```bash
sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh squashfs
```

Background reference:

### Data conversion

Use `mksquashfs` on a source directory:

```bash
mksquashfs <source_directory>/ <output_name>.squashfs
```

Transfer the resulting `.squashfs` file to LUMI (`rsync --partial --progress --stats` is recommended for large files).

### Running PyTorch

Mount squashfs directly into the container and read it as a normal directory:

```bash
singularity exec -B inputs.squashfs:/input-data:image-src=/ mycontainer.sif python my_script.py
```

For image-folder style datasets:

```python
from torchvision.datasets import ImageFolder
dataset = ImageFolder("/train_images")
```

## HDF5

In this repo, use:

```bash
sbatch convert.sh hdf5
```

After conversion finishes, run benchmark:

```bash
sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh hdf5
```

Background reference:

### Data conversion

Conversion can be done fully in Python. For local examples in this repo, see:

- [scripts/hdf5/convert_to_hdf5.py](scripts/hdf5/convert_to_hdf5.py)

### Running PyTorch

Use a custom dataset backed by `h5py`:

- [scripts/hdf5/hdf5_dataset.py](scripts/hdf5/hdf5_dataset.py)
- [scripts/hdf5/visualtransformer-hdf5.py](scripts/hdf5/visualtransformer-hdf5.py)

## LMDB

In this repo, use:

```bash
sbatch convert.sh lmdb
```

After conversion finishes, run benchmark:

```bash
sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh lmdb
```

Background reference:

### Data conversion

Use conversion scripts depending on dataset size/source:

- [scripts/lmdb/convert_to_lmdb.py](scripts/lmdb/convert_to_lmdb.py)
- [scripts/lmdb/convert_large_to_lmdb.py](scripts/lmdb/convert_large_to_lmdb.py)

### Running PyTorch

Use the custom LMDB dataset implementation:

- [scripts/lmdb/lmdb_dataset.py](scripts/lmdb/lmdb_dataset.py)

## Benchmark summary

Synthetic loader benchmarks in this guide showed:

- Tiny ImageNet (100k samples): HDF5 and LMDB similar; SquashFS slower.
- Large ImageNet subset (200k samples): LMDB outperformed SquashFS in this setup.

Reference results from this chapter:

| Tiny dataset | mean (s) | std (s) | N |
| :-- | :--: | :--: | :--: |
| SquashFS | 48.62 | 0.86 | 10 |
| HDF5 | 36.51 | 2.65 | 10 |
| LMDB | 35.26 | 1.98 | 10 |

| Large dataset | mean (s) | std (s) | N |
| :-- | :--: | :--: | :--: |
| SquashFS | 1982.16 | 50.47 | 3 |
| LMDB | 1546.53 | 65.07 | 3 |

## Reproducing benchmarks

- Download tiny raw data first: `bash ./get_data.sh`
- Convert tiny data by format: `sbatch convert.sh <squashfs|lmdb|hdf5>`
- Tiny benchmark: `sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh <squashfs|lmdb|hdf5>`
- Sequential tiny benchmark: `sbatch run-scripts/simple-benchmarks/run-comp-seq.sh <squashfs|lmdb|hdf5>`
- Large benchmark: `sbatch run-scripts/simple-benchmarks/run-comp-large.sh <squashfs|lmdb>`
- Full large benchmark: `sbatch run-scripts/simple-benchmarks/run-comp-large-full.sh <squashfs|lmdb>`
- Dataset-comparison helper: `sbatch run-scripts/dataset-comparison/run_dataset_comp.sh`
- Post-process tiny sequential results: `python3 run-scripts/simple-benchmarks/post-process.py -d seq`
- Post-process tiny parallel results: `python3 run-scripts/simple-benchmarks/post-process.py -d tiny`
- Post-process large results: `python3 run-scripts/simple-benchmarks/post-process.py -d large`
- Post-process full-large results: `python3 run-scripts/simple-benchmarks/post-process.py -d full`

## Verify

Before continuing:

- You can run one benchmark job for your selected format.
- Output contains format-specific loader timing lines.
- You can justify your selected format for your dataset characteristics.

## Troubleshooting

- Conversion overloads filesystem: convert from archive directly when possible, avoid huge raw extraction trees.
- Missing path/runtime variables: source `../env.sh` and verify dataset paths before submission.
- Container mount issues with SquashFS: verify `-B <file>:/mount:image-src=/...` syntax and source path.
- `get_data.sh: Permission denied`: run `bash ./get_data.sh` or set execute permission with `chmod +x get_data.sh`.

## Navigation

- Previous: [2. Setting up your own environment](../2-setting-up-environment/README.md)
- Next: [4. Data Storage Options](../4-data-storage/README.md)
