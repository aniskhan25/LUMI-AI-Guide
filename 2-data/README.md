# 2. Data on LUMI

## Where to store your data

| Filesystem | Use for | Notes |
|---|---|---|
| `/scratch` (LUMI-P) | Active training data | Default choice for most workloads. |
| LUMI-F | I/O-bound workloads | ~10x higher bandwidth than LUMI-P, but 3x the storage cost. |
| `/tmp` (RAMfs) | Node-local fast access | Fastest, but local to one node. Data is lost when the job ends — copy outputs out before exit. |

For more on storage costs, quotas, and optional Lustre striping tuning, see the [LUMI storage documentation](https://docs.lumi-supercomputer.eu/storage/).

## What format to use

| Data type | Recommended format | Why |
|---|---|---|
| Image datasets | **SquashFS** | Mounts directly in the container, no small-file pressure on Lustre |
| Large-scale vision / multimodal | **WebDataset** (tar shards) | Sequential reads, Lustre-friendly, widely used at scale |
| NLP / LLM fine-tuning | **HuggingFace datasets** | De facto standard for text data, Arrow-backed, streaming support |
| Structured / tabular | **HDF5** or **Parquet** | Good for fixed-shape arrays and columnar data |

Avoid storing datasets as millions of loose files on Lustre — this degrades filesystem performance for all users.

## SquashFS demo

SquashFS packs a directory into a single file that Singularity mounts read-only inside the container. No small-file pressure on Lustre during training.

**Step 1** — generate synthetic images and pack them (run on the login node):

```bash
bash prepare_squashfs.sh
```

This writes 100 synthetic images to `$SCRATCH_ROOT/squashfs-demo/images/` and packs them into `demo.squashfs`.

**Step 2** — submit a job that mounts and reads the archive:

```bash
sbatch run_squashfs.sh
```

```bash
cat slurm-<jobid>.out
```

Look for `SQUASHFS READ OK`. In your own training scripts, read from `/data` inside the container as a normal directory.

For WebDataset and HuggingFace datasets formats, see the [LUMI documentation](https://docs.lumi-supercomputer.eu/storage/).

## RAMfs demo

For single-node jobs where your dataset fits in node memory, copying to `/tmp` before training eliminates all Lustre I/O during the run.

```bash
sbatch run_ramfs.sh
```

```bash
cat slurm-<jobid>.out
```

The script copies data into `/tmp`, runs training, then copies the model checkpoint out before the job ends. See `run_ramfs.sh` for the pattern.

## Next

[3. Multi-GPU and Multi-Node Training](../3-multi-gpu-and-node/README.md)
