#!/usr/bin/env bash
#SBATCH --job-name=lattice-s2-ntxent
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage2/ntxent_%j.out
#SBATCH --error=logs/slurm/stage2/ntxent_%j.err
#
# Stage 2 — adapter SSL with NT-Xent / InfoNCE.
#
#   sbatch scripts/slurm/stage2_adapter_ssl_ntxent.sh
#
# Checkpoints: artifacts/adapter/ntxent/checkpoints/<wandb_run_id>/last.ckpt
# Use that path (or the run dir) for Stage 4+ — no export step.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage2_adapter_ssl_ntxent.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

srun python -m lattice_lab.train experiment=adapter_discrete_flow_baseline \
  data.shard_dir=/flash/project_465003063/grassogi/artifacts/processed/moses \
  trainer.max_epochs=10 \
  data.batch_size=64 \
  trainer.accelerator=gpu \
  callbacks.model_checkpoint.dirpath=artifacts/adapter/ntxent/checkpoints
