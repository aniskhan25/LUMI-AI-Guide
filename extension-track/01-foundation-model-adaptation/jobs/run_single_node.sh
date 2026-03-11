#!/bin/bash

#SBATCH --job-name=aif-adapt-1node
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:45:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load lumi-aif-singularity-bindings || module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LESSON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$LESSON_DIR/../.." && pwd)"

source "$REPO_ROOT/env.sh"
: "${CONTAINER:?Set CONTAINER in env.sh}"

RUN_NAME="${RUN_NAME:-baseline-run-1node}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/foundation-adaptation/${RUN_NAME}}"

echo "Lesson directory: $LESSON_DIR"
echo "Output directory: $OUT_DIR"

srun singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
python - <<'PY'
import torch
print(f'GPU_VISIBLE_COUNT={torch.cuda.device_count() if torch.cuda.is_available() else 0}')
PY
cd '$LESSON_DIR'
python data/prepare_sample_data.py --output data/sample_data
python scripts/train.py --config configs/baseline.yaml --output-dir '$OUT_DIR' --run-name '$RUN_NAME'
python scripts/validate_run.py --run-dir '$OUT_DIR' --min-accuracy 0.0
"

