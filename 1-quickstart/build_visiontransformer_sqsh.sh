#!/bin/bash
set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/env.sh"

: "${CONTAINER:?Set CONTAINER in env.sh}"

RESOURCES_DIR="$REPO_ROOT/resources"
OUT_SQSH="$RESOURCES_DIR/visiontransformer-env.sqsh"
BUILD_DIR="$SCRATCH_ROOT/visiontransformer-env-build"

mkdir -p "$RESOURCES_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Building extension environment in: $BUILD_DIR"
singularity exec -B "$BUILD_DIR":/user-software "$CONTAINER" bash -c '
set -euo pipefail
if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi
python -m venv /user-software --system-site-packages
source /user-software/bin/activate
python -m pip install h5py lmdb msgpack six tqdm
'

rm -f "$OUT_SQSH"
echo "Creating squashfs image: $OUT_SQSH"
mksquashfs "$BUILD_DIR" "$OUT_SQSH" -noappend -comp gzip -no-xattrs

echo "Done: $OUT_SQSH"
