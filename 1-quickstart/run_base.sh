#!/bin/bash

#SBATCH --job-name=quickstart-base
#SBATCH --account=project_462000131
#SBATCH --partition=small-g

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/quickstart-base-%j.out

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../scripts/slurm_bootstrap.sh
bootstrap_repo

srun singularity exec "$CONTAINER" bash -c '
  set -euo pipefail
  if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi
  python -c "import platform, torch; print(f\"python={platform.python_version()}\"); print(f\"torch={torch.__version__}\"); print(f\"rocm={torch.version.hip}\"); print(f\"cuda_available={torch.cuda.is_available()}\"); print(f\"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}\"); assert torch.cuda.is_available(), \"GPU not visible in container\"; print(\"SMOKE TEST PASSED\")"
'
