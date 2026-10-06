#!/bin/bash

# Compare the previous and the new LAIF container on 2 nodes (16 GCDs), each under
# its default libfabric memory monitor and under userfaultfd.
# Submit from this directory: sbatch run_bench.sh

#SBATCH --job-name=container-bench
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g

#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=01:00:00

set -euo pipefail

source ../../setup.sh

C=/appl/local/laifs/containers
OLD=$C/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
NEW=$C/lumi-multitorch-u24r72f21m50t211-20260929_104918/lumi-multitorch-full-u24r72f21m50t211-20260929_104918.sif

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

run() {
  local sif=$1 monitor=$2 script=$3
  export FI_MR_CACHE_MONITOR=$monitor
  export MASTER_PORT=$((10000 + RANDOM % 50000))
  echo "=== $(basename "$sif") | FI_MR_CACHE_MONITOR=$monitor | $(basename "$script") ==="
  # A hang under one monitor is itself a result; cap each run so the rest still execute.
  timeout 600 srun --cpu-bind="v,mask_cpu=${CPU_BIND_MASKS}" singularity run "$sif" bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    python $script
  " 2>&1 | grep -v "binding to cpus" || echo "FAILED or TIMED OUT (exit $?)"
}

srun --ntasks-per-node=1 bash -c 'echo "$(hostname): $(ls -l /dev/kdreg2 2>&1)"'

# Old image defaults to memhooks, new image to kdreg2.
for cfg in "$OLD memhooks" "$OLD userfaultfd" "$NEW kdreg2" "$NEW userfaultfd"; do
  set -- $cfg
  run "$1" "$2" allreduce_bench.py
  run "$1" "$2" ../../3-multi-gpu-and-node/visiontransformer_ddp.py
done
