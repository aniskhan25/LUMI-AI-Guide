#!/bin/bash

#SBATCH --job-name=aif-rag
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

RUN_NAME="${RUN_NAME:-rag-baseline}"
OUT_ROOT="${OUT_ROOT:-${SCRATCH_ROOT}/rag-workflows}"

echo "Lesson directory: $LESSON_DIR"
echo "Run name: $RUN_NAME"
echo "Output root: $OUT_ROOT"

srun singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '$LESSON_DIR'
python scripts/prepare_corpus.py --output data
python scripts/chunk_corpus.py --config configs/rag.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/embed_chunks.py --config configs/rag.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/build_index.py --config configs/rag.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/answer_queries.py --config configs/rag.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/validate_rag_run.py --config configs/rag.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME' --require-gpu
"
