#!/bin/bash
# Run on the login node (not as a batch job) to prepare the SquashFS demo:
#
#   bash prepare_squashfs.sh
#
# Generates 100 synthetic images into $SCRATCH_ROOT/squashfs-demo/images,
# then packs them into demo.squashfs in this directory.

set -euo pipefail

source ../setup.sh
cd "$SLURM_SUBMIT_DIR"

DATASET_DIR="${SCRATCH_ROOT}/squashfs-demo/images"
SQUASHFS_FILE="$(pwd)/demo.squashfs"

mkdir -p "$DATASET_DIR"

echo "Generating synthetic images in $DATASET_DIR ..."
singularity exec "$CONTAINER" python - << PY
import os
from torchvision.datasets import FakeData

out_dir = "$DATASET_DIR"
os.makedirs(out_dir, exist_ok=True)

dataset = FakeData(size=100, image_size=(3, 64, 64), num_classes=10)
for i, (img, label) in enumerate(dataset):
    img.save(os.path.join(out_dir, f"img_{i:04d}_class{label}.png"))

print(f"Saved {len(dataset)} images to {out_dir}")
PY

echo "Packing into $SQUASHFS_FILE ..."
mksquashfs "$DATASET_DIR" "$SQUASHFS_FILE" -noappend -no-xattrs

echo "Done. Submit the job with: sbatch run_squashfs.sh"
