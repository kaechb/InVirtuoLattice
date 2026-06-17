#!/usr/bin/env bash
#SBATCH --job-name=lattice-s3-esm2
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage3/%j.out
#SBATCH --error=logs/slurm/stage3/%j.err
#
# Stage 3 — frozen ESM-2 650M protein embeddings (BindingDB + LIT-PCBA targets).
#
#   sbatch scripts/slurm/stage3_protein_precompute.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage3_protein_precompute.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

srun python -m lattice_lab.protein.precompute \
  --fasta artifacts/processed/bindingdb/bindingdb_targets.fasta \
  --store artifacts/protein_store/embeddings/esm2_650M \
  --device cuda \
  --batch-size 8 \
  --no-canonical-filter --overwrite

srun python -m lattice_lab.protein.precompute \
  --fasta artifacts/processed/bindingdb/lit_pcba_targets.fasta \
  --store artifacts/protein_store/embeddings/esm2_650M \
  --device cuda \
  --batch-size 8 \
  --no-canonical-filter --overwrite

