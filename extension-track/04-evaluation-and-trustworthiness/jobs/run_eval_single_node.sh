#!/bin/bash

#SBATCH --job-name=aif-eval
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:00:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load lumi-aif-singularity-bindings || module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LESSON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$LESSON_DIR/../.." && pwd)"

source "$REPO_ROOT/env.sh"
: "${CONTAINER:?Set CONTAINER in env.sh}"

RUN_NAME="${RUN_NAME:-eval-rag-baseline}"
OUT_ROOT="${OUT_ROOT:-${SCRATCH_ROOT}/evaluation-trustworthiness}"

echo "Lesson directory: $LESSON_DIR"
echo "Run name: $RUN_NAME"
echo "Output root: $OUT_ROOT"

srun singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '$LESSON_DIR'
python scripts/run_baseline_eval.py --config configs/eval.yaml --variant baseline --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/run_baseline_eval.py --config configs/eval.yaml --variant candidate --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/score_outputs.py --config configs/eval.yaml --variant baseline --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/score_outputs.py --config configs/eval.yaml --variant candidate --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/extract_failures.py --config configs/eval.yaml --variant baseline --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/extract_failures.py --config configs/eval.yaml --variant candidate --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/compare_variants.py --config configs/eval.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/build_report.py --config configs/eval.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
"

