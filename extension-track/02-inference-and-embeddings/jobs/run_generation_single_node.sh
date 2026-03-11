#!/bin/bash

#SBATCH --job-name=aif-generate
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

RUN_NAME="${RUN_NAME:-generation-baseline}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/inference-generation/${RUN_NAME}}"

echo "Lesson directory: $LESSON_DIR"
echo "Output directory: $OUT_DIR"

srun singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '$LESSON_DIR'
python data/prepare_sample_data.py --output data
python scripts/run_generation.py --config configs/generation.yaml --output-dir '$OUT_DIR' --run-name '$RUN_NAME'
python scripts/validate_outputs.py \
  --mode generation \
  --input-jsonl data/sample_generation_inputs.jsonl \
  --output-jsonl '$OUT_DIR/generation_outputs.jsonl' \
  --summary-json '$OUT_DIR/run_summary.json' \
  --require-gpu
"

