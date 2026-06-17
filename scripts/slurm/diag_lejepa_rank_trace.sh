#!/usr/bin/env bash
#SBATCH --job-name=lattice-diag-lejepa-rank
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/stage2/diag_lejepa_rank_trace_%j.out
#SBATCH --error=logs/slurm/stage2/diag_lejepa_rank_trace_%j.err
#
# One-off diagnostic: same recipe as stage2_adapter_ssl_lejepa.sh (lejepa_lambda
# left at the experiment default, 0.5, to reproduce run 8gcj5i04's dynamics) but
# capped at 4000 steps and with train_rank_every_n_steps=20 so the covariance-
# rank trajectory is visible at fine resolution through the early-training
# window where LeJEPA's invariance-vs-SIGReg tension is sharpest. Throwaway
# checkpoints — do not feed into later stages.
#
#   sbatch scripts/slurm/diag_lejepa_rank_trace.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/diag_lejepa_rank_trace.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

srun python -m lattice_lab.train experiment=adapter_discrete_flow_lejepa \
  data.shard_dir=/flash/project_465003063/grassogi/artifacts/processed/moses \
  trainer.accelerator=gpu \
  trainer.max_steps=4000 \
  model.train_rank_every_n_steps=20 \
  logger.wandb.name=lejepa_rank_trace_diag \
  logger.wandb.group=diag_rank_trace \
  callbacks.model_checkpoint.dirpath=artifacts/adapter/lejepa_diag/checkpoints
