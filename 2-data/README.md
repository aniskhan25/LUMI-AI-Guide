# 2. Data on LUMI

## Where to store your data

| Filesystem | Use for | Notes |
|---|---|---|
| `/scratch` (LUMI-P) | Active training data | Default choice. Set Lustre striping for large files. |
| LUMI-F | I/O-bound workloads | ~10x higher bandwidth than LUMI-P, but 3x the storage cost. |
| `/tmp` (RAMfs) | Node-local fast access | Fastest, but local to one node. Data is lost when the job ends — copy outputs out before exit. |

Set Lustre striping on your data directory before writing large files:

```bash
lfs setstripe --stripe-count 32 --stripe-size 4m /scratch/<project>/<user>/data/
```

For more on storage costs and quotas, see the [LUMI storage documentation](https://docs.lumi-supercomputer.eu/storage/).

## What format to use

| Data type | Recommended format | Why |
|---|---|---|
| Image datasets | **SquashFS** | Mounts directly in the container, no small-file pressure on Lustre |
| Large-scale vision / multimodal | **WebDataset** (tar shards) | Sequential reads, Lustre-friendly, widely used at scale |
| NLP / LLM fine-tuning | **HuggingFace datasets** | De facto standard for text data, Arrow-backed, streaming support |
| Structured / tabular | **HDF5** or **Parquet** | Good for fixed-shape arrays and columnar data |

Avoid storing datasets as millions of loose files on Lustre — this degrades filesystem performance for all users.

### SquashFS

Pack a directory into a single `.squashfs` file, then mount it read-only inside the container:

```bash
mksquashfs /your/dataset/ dataset.squashfs
```

```bash
singularity exec -B dataset.squashfs:/data:image-src=/ "$CONTAINER" python train.py
```

Inside `train.py`, read from `/data` as a normal directory.

### WebDataset

Store samples as tar shards and stream them during training:

```python
import webdataset as wds

dataset = wds.WebDataset("/scratch/<project>/data/shard-{000000..000999}.tar") \
    .decode("pil") \
    .to_tuple("jpg", "cls")
```

Works well on Lustre because reads are large and sequential.

### HuggingFace datasets

```python
from datasets import load_from_disk

dataset = load_from_disk("/scratch/<project>/data/my-dataset")
```

Save once with `dataset.save_to_disk(...)`, reload across jobs. Supports streaming for datasets too large to fit in memory.

## RAMfs example

For single-node jobs where your dataset fits in node memory, copying to `/tmp` before training eliminates all Lustre I/O during the run.

```bash
sbatch run_ramfs.sh
```

The script copies data into `/tmp`, runs training, then copies the model checkpoint out before the job ends. See `run_ramfs.sh` for the pattern.

## Next

[3. Multi-GPU and Multi-Node Training](../5-multi-gpu-and-node/README.md)
