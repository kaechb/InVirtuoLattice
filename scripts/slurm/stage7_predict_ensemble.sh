#!/usr/bin/env bash
#SBATCH --job-name=lattice-s7-predict
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/stage7/%j.out
#SBATCH --error=logs/slurm/stage7/%j.err
#
# Stage 7 — ensemble virtual screening on a compound library.
#
# Reads the variant-namespaced Stage-5 checkpoints
#   artifacts/energy/checkpoints/<run_id>/last.ckpt
#
# Pick the variant + edit the per-seed run ids and the library paths, then:
#   VARIANT=lejepa sbatch scripts/slurm/stage7_predict_ensemble.sh
#   VARIANT=ntxent sbatch scripts/slurm/stage7_predict_ensemble.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage7_predict_ensemble.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

VARIANT="${VARIANT:-lejepa}"
case "${VARIANT}" in
  lejepa) RUN_ID0=5ubbm5ah; RUN_ID1=8o965wj8; RUN_ID2=bv41h3wo ;;
  ntxent) RUN_ID0=gi2762bi; RUN_ID1=uqcvdwg9; RUN_ID2=fn93bboy ;;
  *) echo "unknown VARIANT='${VARIANT}' (use lejepa|ntxent)" >&2; exit 1 ;;
esac
TARGET_FASTA=artifacts/preprocessing/raw/targets/thrb.fasta
TARGET_NAME=THRB
SMILES_FILE=artifacts/preprocessing/raw/libraries/example_library.csv
OUTPUT_CSV=artifacts/predictions/thrb_predictions.csv

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

# The EBM checkpoints carry the full model (frozen adapter + energy head), so the
# adapter is read straight from the first head ckpt — no separate Stage-2 adapter
# to keep in sync.
CKPT0="${REPO}/artifacts/energy/checkpoints/${RUN_ID0}/last.ckpt"
CKPT1="${REPO}/artifacts/energy/checkpoints/${RUN_ID1}/last.ckpt"
CKPT2="${REPO}/artifacts/energy/checkpoints/${RUN_ID2}/last.ckpt"

lattice_require_file "${CKPT0}" "edit the ${VARIANT} seed-0 run id in scripts/slurm/stage7_predict_ensemble.sh"
lattice_require_file "${CKPT1}" "edit the ${VARIANT} seed-1 run id in scripts/slurm/stage7_predict_ensemble.sh"
lattice_require_file "${CKPT2}" "edit the ${VARIANT} seed-2 run id in scripts/slurm/stage7_predict_ensemble.sh"
lattice_require_file "${REPO}/${TARGET_FASTA}" "edit TARGET_FASTA in scripts/slurm/stage7_predict_ensemble.sh"
lattice_require_file "${REPO}/${SMILES_FILE}" "edit SMILES_FILE in scripts/slurm/stage7_predict_ensemble.sh"

mkdir -p "$(dirname "${REPO}/${OUTPUT_CSV}")"

srun python -m lattice_lab.inference.predict_ensemble \
  --head-ckpts \
    "artifacts/energy/checkpoints/${RUN_ID0}/last.ckpt" \
    "artifacts/energy/checkpoints/${RUN_ID1}/last.ckpt" \
    "artifacts/energy/checkpoints/${RUN_ID2}/last.ckpt" \
  --target-fasta "${TARGET_FASTA}" \
  --target-name "${TARGET_NAME}" \
  --smiles-file "${SMILES_FILE}" \
  --n-views 4 \
  --output-csv "${OUTPUT_CSV}"
