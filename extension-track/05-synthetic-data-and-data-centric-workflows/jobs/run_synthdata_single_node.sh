#!/bin/bash

#SBATCH --job-name=aif-synthdata
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:45:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load lumi-aif-singularity-bindings || module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LESSON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$LESSON_DIR/../.." && pwd)"

source "$REPO_ROOT/env.sh"
: "${CONTAINER:?Set CONTAINER in env.sh}"

RUN_NAME="${RUN_NAME:-synthdata-baseline}"
OUT_ROOT="${OUT_ROOT:-${SCRATCH_ROOT}/synthetic-data-workflows}"

echo "Lesson directory: $LESSON_DIR"
echo "Run name: $RUN_NAME"
echo "Output root: $OUT_ROOT"

srun singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '$LESSON_DIR'
python scripts/identify_weak_cases.py --generate-config configs/generate.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/generate_candidates.py --generate-config configs/generate.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/filter_candidates.py --generate-config configs/generate.yaml --filter-config configs/filter.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/merge_augmented_dataset.py --generate-config configs/generate.yaml --filter-config configs/filter.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/rerun_downstream_task.py --generate-config configs/generate.yaml --compare-config configs/compare.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/compare_results.py --generate-config configs/generate.yaml --compare-config configs/compare.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
"

