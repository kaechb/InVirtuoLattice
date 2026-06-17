#!/usr/bin/env bash
#SBATCH --job-name=lattice-diag-hybrid-rank
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/stage2/diag_hybrid_rank_trace_%j.out
#SBATCH --error=logs/slurm/stage2/diag_hybrid_rank_trace_%j.err
#
# One-off diagnostic companion to diag_lejepa_rank_trace.sh: same recipe but
# ssl_loss=hybrid (NT-Xent linearly annealed -> LeJEPA over
# model.hybrid_anneal_steps=2000), with train_rank_every_n_steps=20 so the
# covariance-rank trajectory is visible across the anneal window — does
# effective rank track much higher while ntxent dominates, and does it hold up
# (or crash back down) once lejepa takes over fully? Throwaway checkpoints.
#
#   sbatch scripts/slurm/diag_hybrid_rank_trace.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/diag_hybrid_rank_trace.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

srun python -m lattice_lab.train experiment=adapter_discrete_flow_hybrid \
  data.shard_dir=/flash/project_465003063/grassogi/artifacts/processed/moses \
  trainer.accelerator=gpu \
  trainer.max_steps=4000 \
  model.train_rank_every_n_steps=20 \
  logger.wandb.name=hybrid_rank_trace_diag \
  logger.wandb.group=diag_rank_trace \
  callbacks.model_checkpoint.dirpath=artifacts/adapter/hybrid_diag/checkpoints
