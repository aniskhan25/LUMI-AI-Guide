#!/bin/bash

#SBATCH --job-name=aif-adapt-1gcd
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load lumi-aif-singularity-bindings || module load singularity-AI-bindings

LESSON_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(cd -- "$LESSON_DIR/../.." && pwd)"

source "$REPO_ROOT/env.sh"
: "${CONTAINER:?Set CONTAINER in env.sh}"

RUN_NAME="${RUN_NAME:-baseline-run}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/foundation-adaptation/${RUN_NAME}}"

echo "Lesson directory: $LESSON_DIR"
echo "Output directory: $OUT_DIR"

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

srun singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '$LESSON_DIR'
python data/prepare_sample_data.py --output data/sample_data
python scripts/train.py --config configs/baseline.yaml --output-dir '$OUT_DIR' --run-name '$RUN_NAME'
python scripts/validate_run.py --run-dir '$OUT_DIR' --min-accuracy 0.0
"
