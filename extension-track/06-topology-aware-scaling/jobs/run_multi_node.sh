#!/bin/bash

#SBATCH --job-name=scale-multinode
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:40:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load lumi-aif-singularity-bindings || module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LESSON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$LESSON_DIR/../.." && pwd)"

source "$REPO_ROOT/env.sh"
: "${CONTAINER:?Set CONTAINER in env.sh}"

OUT_ROOT="${OUT_ROOT:-${SCRATCH_ROOT}/topology-scaling}"
RUN_NAME="${RUN_NAME:-scaling-multi-node}"

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

echo "Lesson directory: $LESSON_DIR"
echo "Output root: $OUT_ROOT"
echo "Run name: $RUN_NAME"
echo "MASTER_ADDR: $MASTER_ADDR"

srun --cpu-bind=cores --distribution=block:block singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '$LESSON_DIR'
python -m torch.distributed.run \
  --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 \
  --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint='$MASTER_ADDR:$MASTER_PORT' \
  scripts/inspect_placement.py --config configs/multi_node.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python -m torch.distributed.run \
  --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 \
  --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint='$MASTER_ADDR:$MASTER_PORT' \
  scripts/run_workload.py --config configs/multi_node.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
python scripts/collect_metrics.py --config configs/multi_node.yaml --output-root '$OUT_ROOT' --run-name '$RUN_NAME'
"

