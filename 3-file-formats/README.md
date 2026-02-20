# 3. File formats for training data

## Goal

Understand tradeoffs between SquashFS, HDF5, and LMDB on LUMI, and choose a practical format for your training data pipeline.

## Assumptions

- You can run Python inside a LUMI container and install extra packages when needed.
- You completed [2. Setting up your own environment](../2-setting-up-environment/README.md).
- The Python and shell scripts in `3-file-formats/` are available if you want to reproduce the benchmarks.

## What changes from baseline

- Baseline you already have: a runnable training environment and working dataset path conventions.
- This lesson adds: format selection and conversion strategy for SquashFS, HDF5, and LMDB.
- Expected output/artifact: a chosen training-data format plus converted dataset artifacts and benchmark evidence.

## Minimal run checkpoint

Command:

```bash
sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh lmdb
```

Success signal:

- Job output contains `LMDB dataloader time:` (or the equivalent `* dataloader time:` line for your selected format).

## Introduction

Generally, there is no one-size-fits-all file format suitable for all machine learning and artificial intelligence data. Different high-performance file formats have different strategies for increasing the read/write throughput, and these strategies might not be compatible with the format of the data (e.g. variable image sizes). As a result, this compatibility must be determined before an optimal file format can be chosen. 

Another practical issue is the data conversion necessary to change one's data from its current file format to the desired target file format. This is primarily an issue for large datasets containing hundreds of thousands of small files on a parallel file system like on LUMI. Converting the data to a raw format must be avoided at all costs to preserve the integrity of the file system for all users. This issue can be circumvented in various ways, one option is to prepare the data in the desired file format before it is transferred to LUMI, another option is to convert directly from the initial format (often .zip) to the target file format.

After converting the dataset to the desired file format, it must also be efficiently processed in PyTorch. Different formats have different requirements here as well, where some require writing custom classes, while others are plug-and-play. Custom datasets can be built in PyTorch using the [built-in base classes](https://pytorch.org/vision/stable/datasets.html#base-classes-for-custom-datasets). 

Finally, the actual performance of the various file formats is the final deciding factor. The performance of the different file format has been analyzed with a 5GB tiny imagenet and 157GB full imagenet. For the tiny imagenet we found near identical performance for all file formats.

## Squashfs
Squashfs is perhaps the simplest way to get started with a PyTorch AI/ML workflow on a HPC platform. It poses no restrictions in terms of compatibility, and it requires the least amount of custom data parsers and scripts out of the formats tested here. However, it does currently require a local linux system for data conversion. It is also the least performing option on large datasets.

### Data conversion
Data conversion is done using the command `mksquashfs` available in various Linux package managers (`apt-get`, `dnf`, `zypper`, ...). If we have a raw data folder `ILSVRC/` for imagenet, we can convert it to the squashfs file format using 
```bash
mksquashfs ILSVRC/ imagenet.squashfs
```
Then the `.squashfs` file is ready to be transferred to LUMI. This can be done in a variety of ways as seen in the [LUMI documentation](https://docs.lumi-supercomputer.eu/firststeps/movingdata/). For this purpose `rsync` is particularly useful. Use the flags `--partial` and `--progress --stats` to preserve partial transfers and display detailed progress statistics for the large file. 

### Running PyTorch
Running PyTorch with data stored in the `squashfs` file format is particularly simple because we are already utilizing containers which were introduced in chapter [Setting up your own environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/2-setting-up-environment#readme). The singularity container supports [mounting](https://docs.sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html#squashfs-image-files) the `squashfs` file directly into the file system when running the container,
```bash
singularity run -B inputs.squashfs:/input-data:image-src=/ mycontainer.sif
```
where `inputs.squashfs` is the relative path to the stored `.squashfs` file, `:/input-data` is the folder where the data will appear inside the container file system and finally `:image-src=/` tells singularity it should be mounted as a folder (in contrast to a single file), and the `/` describes the path inside the `.squashfs` file the mount will appear. 
 For example, if the folder `ILSVRC/` has a deep tree structure such as `ILSVRC/Data/CLS-LOC/train/...`, where I only need the training data in the `train/...` folder, the bind-mount would be
```bash
 -B scratch/project_465XXXXXX/data/imagenet.squashfs:/train_images:image-src=/Data/CLS-LOC/train/
```
where the large squashfs data is stored in the project's scratch folder `scratch/project_465XXXXXX/data`. We can then run PyTorch using the built-in dataset `ImageFolder` as if the dataset was stored in an ordinary folder inside the container,
```python
from torchvision.datasets import ImageFolder
dataset = ImageFolder('/train_images')  # Data is bind-mounted at /train_images 
```

## HDF5
Hierarchical Data Format (HDF5) is a well-established high-performance file format, which interfaces well with the popular `numpy` library through its `h5py` Python interface. This convenience does come at a cost of poor compatibility with irregularly shaped data, such as images with varying shapes and graph networks. 

### Data conversion
Converting a dataset into the HDF5 format can be done entirely in Python. In order to unpack archive data on LUMI, you first need a parser that can stream the archive into HDF5 using packages such as `zipfile`, `tarfile`, or `shutil`. Once the parser is in place, creating the target HDF5 file is straightforward with `h5py.File().create_dataset`; see [convert_to_hdf5.py](scripts/hdf5/convert_to_hdf5.py) for a concrete example.

### Running PyTorch
To create a PyTorch `DataLoader`, you need a custom `Dataset` implementation that knows dataset size and item retrieval. For HDF5, this is typically done by opening the file via `h5py` and indexing numpy-like objects. See [hdf5_dataset.py](scripts/hdf5/hdf5_dataset.py) and its usage in [visualtransformer-hdf5.py](scripts/hdf5/visualtransformer-hdf5.py).

## LMDB
Lightning Memory-Mapped Database (LMDB) is a very fast file format, which like squashfs imposes no restriction on the shape of the data and thus offers good compatibility. This is achieved through memory-mapped files that provide fast access without necessarily loading the entire dataset into memory.

### Data conversion
The conversion process is similar to HDF5: parse archive contents and write to LMDB using the `lmdb` Python library. Because LMDB is more flexible than ndarray-style formats, write/read code is usually a bit more custom. See [convert_to_lmdb.py](scripts/lmdb/convert_to_lmdb.py) for tiny ImageNet and [convert_large_to_lmdb.py](scripts/lmdb/convert_large_to_lmdb.py) for the large archive case.

### Running PyTorch
As with HDF5, we use a custom `Dataset` for LMDB to load data efficiently through PyTorch `DataLoader`; see [lmdb_dataset.py](scripts/lmdb/lmdb_dataset.py). The implementation is typically more involved because values are encoded and decoded from binary payloads.

## Performance
The following benchmarks compare loading performance across the different file formats using PyTorch DataLoader on both small and large ImageNet datasets.

### Synthetic Benchmark
In the synthetic benchmark, we measure how quickly samples can be loaded into Python using the PyTorch `DataLoader` for the various different file formats. The loop time is measured for both the tiny and large ImageNet a number of times. Here we report the measured average and standard deviation. 
For the tiny imagenet, we loop through the entire dataset of 100.000 images. This is tested `N` times, where each job is executed independently to ensure a fresh node is used each time. The result is as follows;

|          | mean (s) | std (s) |  N  |
| :------: | :------: | :-----: | :-: |
| squashfs |  48.62   |  0.86   | 10  |
|   HDF5   |  36.51   |  2.65   | 10  |
|   LMDB   |  35.26   |  1.98   | 10  |

HDF5 and LMDB show similar performance, while squashfs is about 33% slower. The parameters of the `DataLoader` are as follows:
`DataLoader(data, batch_size=32, shuffle=True, num_workers=7)`
That is, the data is shuffled to be loaded in a random order, and is loaded in batches of 32 samples at a time. The number of workers is set equal to the number of CPUs requested in the allocation. Where on [LUMI one should maximally request 7 cores per GPU requested](https://lumi-supercomputer.github.io/LUMI-training-materials/User-Updates/Update-202308/responsible-use/#core-and-memory-use-on-small-g-and-dev-g).

We can repeat the benchmark in a sequential job with one CPU core and `num_workers=1`: we find that squashfs and LMDB scales as you would expect, however HDF5 does not run well sequentially.

|          | mean (s) | std (s) |  N  |
| :------: | :------: | :-----: | :-: |
| squashfs |  247.25  |  1.53   |  5  |
|   HDF5   |  1884.7  |  0.46   |  5  |
|   LMDB   |  209.95  |  15.99  |  5  |

For the large imagenet, we loop through 200.000 out of the 1.2 million images for the formats compatible with varying image size. The varying image sizes pose a critical problem for the HDF5 format, since it requires the data to fit into `ndarray`-like (d-dimensional hypercube) data structures. While data padding is possible, this is not pursued here to keep the comparison fair. The job is again executed independently `N` times and identical `DataLoader` parameters are used.

|          | mean (s) | std (s) |  N  |
| :------: | :------: | :-----: | :-: |
| squashfs | 1982.16  |  50.47  |  3  |
|   LMDB   | 1546.53  |  65.07  |  3  |

LMDB shows roughly 28% better performance than squashfs. 

## Verify

Before continuing, check that:

- You can describe when each format is a better fit (compatibility, conversion cost, and throughput).
- You can run one data conversion workflow relevant to your dataset.
- Your DataLoader runs successfully with your selected format.

## Troubleshooting

- Conversion is too slow or stresses Lustre: avoid unpacking huge raw trees on LUMI; prefer direct conversion from archives.
- Data loading errors with custom datasets: verify your parser/dataset class handles labels and binary payloads correctly.
- Container mount issues with SquashFS: confirm bind path and mount syntax match your image and target directory.

## Navigation

- Previous: [2. Setting up your own environment](../2-setting-up-environment/README.md)
- Next: [4. Data Storage Options](../4-data-storage/README.md)
