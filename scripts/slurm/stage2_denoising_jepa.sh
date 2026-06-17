#!/usr/bin/env bash
#SBATCH --job-name=lattice-s2-djepa
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage2/djepa_%j.out
#SBATCH --error=logs/slurm/stage2/djepa_%j.err
#
# Stage 2 — conditional denoising-JEPA SSL on the discrete-flow (DDiT) backbone.
# An encoder attention-pools the clean SMILES into a molecule latent z_s; z_s
# conditions a separate DDiT denoiser that reconstructs a corrupted copy with
# pure token CE at the noised positions (generative loss, no EMA teacher).
# A Tanimoto fingerprint distillation term (model.fp_weight) gives z_s chemical
# structure so the generative loss does not erode val/probe_r2_*.
#
#   sbatch scripts/slurm/stage2_denoising_jepa.sh
#
# Checkpoints: artifacts/adapter/denoising_jepa/checkpoints/<wandb_run_id>/last.ckpt
# Use that path (or the run dir) for Stage 4+ — no export step.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage2_denoising_jepa.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu


srun python -m lattice_lab.train experiment=denoising_jepa \
  data.shard_dir=/flash/project_465003063/grassogi/artifacts/processed/moses \
  trainer.max_epochs=10 \
  trainer.accelerator=gpu \
  callbacks.model_checkpoint.dirpath=artifacts/adapter/denoising_jepa/checkpoints \
  