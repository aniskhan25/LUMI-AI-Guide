# 1. QuickStart

This chapter covers how to set up the environment to run the [`visiontransformer.py`](visiontransformer.py) script on LUMI. 

First, you clone this repository to LUMI via the following command:

```bash
git clone https://github.com/Lumi-supercomputer/LUMI-AI-Guide.git
```

We recommend using your `/project/` or `/scratch/` directory of your project to clone the repository as your home directory (`$HOME`) has a capacity of 20 GB and is intended to store user configuration files and personal data.

Next, navigate to the `LUMI-AI-Guide/1-quickstart` directory:

```bash
cd LUMI-AI-Guide/1-quickstart
```

The recommended quickstart flow has three steps:

1. Smoke-test the base container.
2. Build a squashfs extension with missing Python packages.
3. Run the Vision Transformer training script with that extension.

This keeps the runtime model consistent with the rest of the guide.

## Step 1: Smoke test the base container

Submit the base-container smoke test:

```bash
sbatch run_base.sh
```

Check the job output in:

```bash
/scratch/<project>/<user>/slurm/quickstart-base-<jobid>.out
```

You should see Python/Torch/ROCm version info and `SMOKE TEST PASSED`.

## Step 2: Build the squashfs extension

Build `visiontransformer-env.sqsh` from the base container:

```bash
./build_visiontransformer_sqsh.sh
```

This writes:

- `resources/visiontransformer-env.sqsh`

The extension includes packages needed by the sample scripts, such as `h5py`.

## Step 3: Run Vision Transformer

Submit:

```bash
sbatch run.sh
```

`run.sh` uses:

- `env.sh` for container selection
- `scripts/slurm_bootstrap.sh` for repo/env setup
- `resources/visiontransformer-env.sqsh` for the extended Python environment

For this example, we use the [Tiny ImageNet Dataset](https://paperswithcode.com/dataset/tiny-imagenet) which is already transformed into the file system friendly hdf5 format (Chapter [File formats for training data](../3-file-formats/README.md) explains in detail why this step is necessary). Please have a look at the terms of access for the ImageNet Dataset [here](https://www.image-net.org/download.php).

To run the Vision Transformer example, we use the batch job script [`run.sh`](run.sh), which runs [`visiontransformer.py`](visiontransformer.py) on a single GPU on a LUMI-G node.
A quickstart to SLURM is provided in the [LUMI documentation](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/). 

If needed, replace the `--account` flag in [`run_base.sh`](run_base.sh) and [`run.sh`](run.sh) with your own project account. You can find your project account by running `lumi-workspaces`.

Once the job starts, output is written to:

```bash
/scratch/<project>/<user>/slurm/quickstart-<jobid>.out
```

The output will show Loss and Accuracy values for each epoch, similar to the following:

```bash
Epoch 1, Loss: 4.68622251625061
Accuracy: 9.57%
Epoch 2, Loss: 4.104039922332763
Accuracy: 15.795%
Epoch 3, Loss: 3.7419378942489625
Accuracy: 19.525%
Epoch 4, Loss: 3.6926351853370667
Accuracy: 21.265%
...
```

Congratulations! You have run your first training job on LUMI. The next chapter [Setting up your own environment](../2-setting-up-environment/README.md) explains in more detail how to build and maintain your own environment.

 ### Table of contents

- [Home](..#readme)
- [1. QuickStart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/1-quickstart#readme)
- [2. Setting up your own environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/2-setting-up-environment#readme)
- [3. File formats for training data](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/3-file-formats#readme)
- [4. Data Storage Options](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/4-data-storage#readme)
- [5. Multi-GPU and Multi-Node Training](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/5-multi-gpu-and-node#readme)
- [6. Monitoring and Profiling jobs](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/6-monitoring-and-profiling#readme)
- [7. TensorBoard visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/7-TensorBoard-visualization#readme)
- [8. MLflow visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/8-MLflow-visualization#readme)
- [9. Wandb visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/9-Wandb-visualization#readme)
