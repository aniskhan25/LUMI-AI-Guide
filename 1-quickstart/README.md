# 1. QuickStart

This lesson gives you a minimal, end-to-end first run of [`visiontransformer.py`](visiontransformer.py) on LUMI.

## Goal

Run one single-GPU training job on LUMI and confirm that:

- the container works on GPU
- the squashfs extension is built
- training logs and a model checkpoint are produced

## Assumptions

- You have a LUMI project account and can submit jobs to `small-g`.
- The repository is cloned on `/project` or `/scratch` (not `$HOME`).
- `../env.sh` is configured for your environment and points to a valid container via `CONTAINER`.
- `../resources/train_images.hdf5` is available.

Clone this repository to LUMI if needed:

```bash
git clone https://github.com/Lumi-supercomputer/LUMI-AI-Guide.git
```

Use `/project` or `/scratch` for this clone. `$HOME` has limited capacity and is meant for configuration and personal files.

Then move to the lesson directory:

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

## Verify

After the three steps, confirm all of the following:

- Base smoke-test output includes `SMOKE TEST PASSED`.
- `../resources/visiontransformer-env.sqsh` exists.
- Training job output includes epoch-level loss and accuracy logs.
- `vit_b_16_imagenet.pth` is created in `1-quickstart/`.

## Troubleshooting

- Job fails at submission: update `--account` in [`run_base.sh`](run_base.sh) and [`run.sh`](run.sh), then check with `lumi-workspaces`.
- Container variable error (`Set CONTAINER in env.sh`): set `CONTAINER` in `../env.sh` to a valid `.sif`.
- Missing dataset error: place or point `TINY_HDF5_PATH` to `../resources/train_images.hdf5`.
- No GPU visible in smoke test: ensure `module load singularity-AI-bindings` is present and rerun.

## Navigation

- Previous: [0. How to use this guide](../0-how-to-use-guide/README.md)
- Next: [2. Setting up your own environment](../2-setting-up-environment/README.md)
