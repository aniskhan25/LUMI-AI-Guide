#!/bin/bash

#SBATCH --job-name=aif-infer-batched
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load lumi-aif-singularity-bindings || module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LESSON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$LESSON_DIR/../.." && pwd)"

source "$REPO_ROOT/env.sh"
: "${CONTAINER:?Set CONTAINER in env.sh}"

OUT_ROOT="${OUT_ROOT:-${SCRATCH_ROOT}/advanced-inference-serving}"
RUN_NAME="${RUN_NAME:-advanced-inference-batched}"

echo "Lesson directory: $LESSON_DIR"
echo "Output root: $OUT_ROOT"
echo "Run name: $RUN_NAME"

srun singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '$LESSON_DIR'
python scripts/run_batched_inference.py --config configs/inference.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/collect_metrics.py --config configs/inference.yaml --mode batched --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
"

