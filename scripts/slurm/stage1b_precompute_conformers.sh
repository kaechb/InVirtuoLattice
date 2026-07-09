#!/usr/bin/env bash
#SBATCH --job-name=lattice-s1b-conf
#SBATCH --account=project_465003063
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage1b/%j.out
#SBATCH --error=logs/slurm/stage1b/%j.err
#
# Stage 1b — precompute one RDKit ETKDGv3 conformer per MOSES molecule and cache
# it to conformers.parquet (consumed by the model.encoder_3d cross-modal view).
# CPU-only; run after Stage 1 has written the fragment-view shards.
#
#   sbatch scripts/slurm/stage1b_precompute_conformers.sh
#   MERGE=1 sbatch scripts/slurm/stage1b_precompute_conformers.sh   # _merge shards
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage1b_precompute_conformers.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_load_cpu_modules
lattice_cd_repo

MERGE="${MERGE:-0}"
case "${MERGE}" in
  0|false|no|"") SUFFIX="" ;;
  1|true|yes)    SUFFIX="_merge" ;;
  *) echo "MERGE=${MERGE} (want 0 or 1)" >&2; exit 1 ;;
esac

SHARD_DIR="${REPO}/artifacts/preprocessing/processed/moses${SUFFIX}"
OUTPUT="${SHARD_DIR}/conformers.parquet"
OVERWRITE="${OVERWRITE:-1}"

# Idempotent when driven by the pipeline (OVERWRITE=0): skip if the cache exists.
if [[ -f "${OUTPUT}" && "${OVERWRITE}" != 1 ]]; then
  lattice_job_banner "conformers cache exists (${OUTPUT}); skip (OVERWRITE=1 to rebuild)"
  exit 0
fi
OVERWRITE_ARGS=()
[[ "${OVERWRITE}" == 1 ]] && OVERWRITE_ARGS=(--overwrite)

srun python -m lattice_lab.preprocessing.precompute_conformers \
  --shard-dir "${SHARD_DIR}" \
  --output "${OUTPUT}" \
  --n-jobs 64 \
  "${OVERWRITE_ARGS[@]}"
