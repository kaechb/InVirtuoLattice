#!/usr/bin/env bash
#SBATCH --job-name=lattice-s5-lejepa
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/slurm/stage5/lejepa_%A_%a.out
#SBATCH --error=logs/slurm/stage5/lejepa_%A_%a.err
#
# Stage 5 — EBM training on LeJEPA decoy pools (one seed per array task).
#
#   sbatch scripts/slurm/stage5_ebm_train_lejepa.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage5_ebm_train_lejepa.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

SEED="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

srun python -m lattice_lab.train experiment=ebm_hardneg_lejepa \
  trainer.max_steps=12000 \
  seed="${SEED}" \
  callbacks.model_checkpoint.dirpath="artifacts/energy/lejepa"
