#!/usr/bin/env bash
#SBATCH --job-name=lattice-s2-lejepa
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage2/lejepa_%j.out
#SBATCH --error=logs/slurm/stage2/lejepa_%j.err
#
# Stage 2 — adapter SSL with LeJEPA (invariance + SIGReg).
#
#   sbatch scripts/slurm/stage2_adapter_ssl_lejepa.sh
#
# Checkpoints: artifacts/adapter/lejepa/checkpoints/<wandb_run_id>/last.ckpt
# Use that path (or the run dir) for Stage 4+ — no export step.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage2_adapter_ssl_lejepa.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

srun python -m lattice_lab.train experiment=adapter_discrete_flow_lejepa \
  data.shard_dir=/flash/project_465003063/grassogi/artifacts/processed/moses \
  trainer.max_epochs=10 \
  trainer.accelerator=gpu \
  callbacks.model_checkpoint.dirpath=artifacts/adapter/lejepa/checkpoints model.lejepa_lambda=0.01 logger.name=2global-4local model.lejepa_n_global_views=2 model.lejepa_n_local_views=4
