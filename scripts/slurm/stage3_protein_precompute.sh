#!/usr/bin/env bash
#SBATCH --job-name=lattice-s3-protein
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
# Stage 3 — frozen protein embeddings (BindingDB + LIT-PCBA targets).
#
#   sbatch scripts/slurm/stage3_protein_precompute.sh
#   PROTEIN=esmc sbatch scripts/slurm/stage3_protein_precompute.sh
#   OVERWRITE=0 sbatch scripts/slurm/stage3_protein_precompute.sh   # incremental (pipeline default)
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage3_protein_precompute.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_protein_resolve
# ponytail: pipeline sets OVERWRITE=0; standalone default rebuilds the store.
OVERWRITE="${OVERWRITE:-1}"

OVERWRITE_ARGS=()
[[ "${OVERWRITE}" == 1 ]] && OVERWRITE_ARGS=(--overwrite)

lattice_load_gpu_modules
lattice_cd_repo
lattice_pipeline_track_slurm_logs 3
lattice_require_gpu

export WANDB_MODE=disabled

for _fasta in bindingdb_targets.fasta lit_pcba_targets.fasta; do
  srun python -m lattice_lab.protein.precompute \
    "${PROTEIN_EXTRA[@]}" \
    --fasta "artifacts/preprocessing/processed/bindingdb/${_fasta}" \
    --store "${PROTEIN_STORE}" \
    --device cuda \
    --batch-size 8 \
    "${OVERWRITE_ARGS[@]}"
done
