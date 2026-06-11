#!/bin/bash

#SBATCH --job-name=aif-rag
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

source ../../env.sh
: "${CONTAINER:?Set CONTAINER in env.sh}"

singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
python scripts/chunk_corpus.py --config configs/rag.yaml
python scripts/embed_chunks.py --config configs/rag.yaml
python scripts/build_index.py --config configs/rag.yaml
python scripts/answer_queries.py --config configs/rag.yaml
"
