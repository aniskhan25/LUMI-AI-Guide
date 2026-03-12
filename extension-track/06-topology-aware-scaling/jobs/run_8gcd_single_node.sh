#!/bin/bash

#SBATCH --job-name=scale-8gcd
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:30:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load lumi-aif-singularity-bindings || module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LESSON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$LESSON_DIR/../.." && pwd)"

source "$REPO_ROOT/env.sh"
: "${CONTAINER:?Set CONTAINER in env.sh}"

OUT_ROOT="${OUT_ROOT:-${SCRATCH_ROOT}/topology-scaling}"
RUN_NAME="${RUN_NAME:-scaling-8gcd-single-node}"

echo "Lesson directory: $LESSON_DIR"
echo "Output root: $OUT_ROOT"
echo "Run name: $RUN_NAME"

srun --cpu-bind=cores --distribution=block:block singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '$LESSON_DIR'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/inspect_placement.py \
  --config configs/single_node.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/run_workload.py \
  --config configs/single_node.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/collect_metrics.py --config configs/single_node.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
"

